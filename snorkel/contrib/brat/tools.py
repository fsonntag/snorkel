import os
import re
import sys
import glob
import codecs
from sqlalchemy.sql import select
from collections import defaultdict
from ...db_helpers import reload_annotator_labels
from ...parser import TextDocPreprocessor, CorpusParser
from ...models import Candidate, StableLabel, Document, TemporarySpan, Sentence, candidate_subclass, GoldLabel

from tqdm import tqdm


class BratProject(object):
    """
    Snorkel Import/Export for
    Brat Rapid Annotation Tool
    http://brat.nlplab.org/

    Brat uses standoff annotation format (see: http://brat.nlplab.org/standoff.html)

    Annotation ID Types
    T: text-bound annotation
    R: relation
    E: event
    A: attribute
    M: modification (alias for attribute, for backward compatibility)
    N: normalization [new in v1.3]
    #: note

    Many of of the advanced schema abilities of BRAT are not implemented, so
    mind the following caveats:

    (1) We do not currently support hierarchical entity definitions, e.g.,
            !Anatomical_entity
                !Anatomical_structure
                    Organism_subdivision
                    Anatomical_system
                    Organ
    (2) All relations must be binary with a single argument type
    (3) Attributes, normalization, and notes are added as candidate meta information

    """

    TEXT_BOUND_ID = 'T'
    RELATION_ID = 'R'
    EVENT_ID = 'E'
    ATTRIB_ID = 'A'
    MOD_ID = 'M'
    NORM_ID = 'N'
    NOTE_ID = '#'

    def __init__(self, session, tmpl_path='annotation.config.tmpl', encoding="utf-8", verbose=True):
        """
        Initialize BRAT import tools.
        :param session:
        :param tmpl_path:    annotation config template. don't change this.
        :param encoding:
        :param verbose:
        """
        self.session = session
        self.encoding = encoding
        self.verbose = verbose

        # load brat config template
        mod_path = "{}/{}".format(os.path.abspath(os.path.dirname(__file__)), tmpl_path)
        self.brat_tmpl = "".join(open(mod_path, "rU").readlines())

        # snorkel dynamic types
        self.subclasses = {}

    def import_project(self, input_dir, annotations_only=True, annotator_name='brat', num_threads=1, parser=None):
        """
        Import BART project,
        :param input_dir:
        :param autoreload:
        :param num_threads:
        :param parser:
        :return:
        """
        config_path = "{}/{}".format(input_dir, "annotation.conf")
        if not os.path.exists(config_path):
            print("Fatal error: missing 'annotation.conf' file", file=sys.stderr)
            return

        # load brat config (this defines relation and argument types)
        config = self._parse_config(config_path)
        anno_filelist = set([os.path.basename(fn).strip(".ann") for fn in glob.glob(input_dir + "/*.ann")])

        # import standoff annotations for all documents
        annotations = {}
        for fn in anno_filelist:
            txt_fn = "{}/{}.txt".format(input_dir, fn)
            ann_fn = "{}/{}.ann".format(input_dir, fn)
            if os.path.exists(txt_fn) and os.path.exists(ann_fn):
                annotations[fn] = self._parse_annotations(txt_fn, ann_fn)

        # by default, we parse and import all project documents
        if not annotations_only:
            self._parse_documents(input_dir + "/*.txt", num_threads, parser)

        # create types
        self._create_candidate_subclasses(config)

        # create candidates
        self._create_candidates(annotations, annotator_name)

    def export_project(self, output_dir, positive_only_labels=True):
        """

        :param output_dir:
        :return:
        """
        candidates = self.session.query(Candidate).all()
        documents = self.session.query(Document).all()

        gold_labels = {label.candidate_id: label for label in self.session.query(GoldLabel).all()}
        gold_labels = {uid:label for uid, label in gold_labels.items()
                      if (positive_only_labels and label.value == 1) or not positive_only_labels}

        doc_index     = {doc.name:doc for doc in documents}
        cand_index    = _group_by_document(candidates)
        snorkel_types = {type(c): 1 for c in candidates}

        for name in doc_index:
            doc_anno = self._build_doc_annotations(cand_index[name], gold_labels) if name in cand_index else []
            fname = "{}{}".format(output_dir,name)
            #  write .ann files
            with codecs.open(fname + ".ann",'w',self.encoding) as fp:
                fp.write("\n".join(doc_anno))
            # write documents
            with codecs.open(fname + ".txt",'w',self.encoding) as fp:
                fp.write(doc_to_text(doc_index[name]))

        # export config file
        config = self._create_config(snorkel_types)
        config_path = "{}annotation.conf".format(output_dir)
        with codecs.open(config_path, 'w', self.encoding) as fp:
            fp.write(config)

        if self.verbose:
            print("Export complete")
            print("\t {} documents".format(len(doc_index)))
            print("\t {} annotations".format( sum([len(cand_index[name]) for name in cand_index] )))

    def _get_arg_type(self, c, span, use_titlecase=True):
        """
        Given a span object, determine it's internal type
        TODO: What is a better way of doing this?

        :param c:
        :param span:
        :param use_titlecase:
        :return:
        """
        for key in c.__dict__.keys():
            if c.__dict__[key] == span:
                key = map(lambda x:x[0].upper()+x[1:], re.split("[-_]",key))
                return "".join(key)
        return None

    def _get_normed_rela_name(self, name):

        name = re.split("[-_]", name)
        if len(name) == 1:
            return name[0]
        name = map(lambda x: x.lower(), name)
        return "".join(map(lambda x: x[0].upper() + x[1:], name))

    def _build_doc_annotations(self, cands, gold_labels=[]):
        """
        Assume binary relation defs

        :param cands:
        :return:
        """
        entities,relations,types = {},{},{}
        for i,c in enumerate(cands):
            if c.id not in gold_labels:
                continue
            for span in c:
                if span not in entities:
                    types[span] = self._get_arg_type(c,span)
                    entities[span] = ("T",len(entities)+1)
            arg1 = "{}{}".format(*entities[c[0]])
            arg2 = "{}{}".format(*entities[c[1]])
            relations[('R',len(relations)+1)] =  "{} Arg1:{} Arg2:{}".format(type(c).__name__, arg1, arg2)

        entities = {uid:span for span,uid in entities.items()}
        annotations = []
        # export entities (relation arguments)
        for uid in sorted(entities, key=lambda x:x[-1]):
            span = entities[uid]
            char_start, char_end = map(int,span.stable_id.split(":")[-2:])
            char_end += 1
            arg_id = "{}{}".format(*uid)
            annotations.append("{}\t{} {} {}\t{}".format(arg_id, types[span], char_start, char_end, span.get_span()))

        # export relations
        for uid in sorted(relations, key=lambda x:x[-1]):
            arg_id = "{}{}".format(*uid)
            annotations.append("{}\t{}".format(arg_id, relations[uid]))

        # candidate attributes
        # TODO

        return annotations

    def _parse_documents(self, input_path, num_threads, parser):
        """

        :param input_path:
        :param num_threads:
        :param parser:
        :return:
        """
        doc_preprocessor = TextDocPreprocessor(path=input_path, encoding=self.encoding)
        corpus_parser = CorpusParser(parser)
        corpus_parser.apply(doc_preprocessor, parallelism=num_threads)

    def _parse_annotations(self, txt_filename, ann_filename):
        """
        Use parser to import BRAT backoff format
        TODO: Currently only supports Entities & Relations

        :param txt_filename:
        :param ann_filename:
        :return:
        """
        annotations = {}

        # load document
        doc = []
        with codecs.open(txt_filename, "rU", encoding=self.encoding) as fp:
            for line in fp:
                doc += [line.strip().split()]

        # build doc string and char to word index
        doc_str = ""
        char_idx = {}
        for i, sent in enumerate(doc):
            for j in range(0, len(sent)):
                char_idx[len(doc_str)] = (i, j)
                for ch in sent[j]:
                    doc_str += ch
                    char_idx[len(doc_str)] = (i, j)
                doc_str += " " if j != len(sent) - 1 else "\n"
        doc_str = doc_str.strip()

        # load annotations
        with codecs.open(ann_filename, "rU", encoding=self.encoding) as fp:
            for line in fp:
                row = line.strip().split("\t")
                anno_id_prefix = row[0][0]
                # parse each entity/relation type
                if anno_id_prefix == Brat.TEXT_BOUND_ID:
                    anno_id, entity, text = row
                    entity_type = entity.split()[0]
                    spans = map(lambda x: map(int, x.split()),
                                entity.lstrip(entity_type).split(";"))

                    # discontinuous mentions
                    if len(spans) != 1:
                        print>> sys.stderr, "NotImplementedError: Discontinuous Spans"
                        continue

                    entity = []
                    for (i, j) in spans:
                        if i in char_idx:
                            mention = doc_str[i:j]
                            tokens = mention.split()
                            sent_id, word_offset = char_idx[i]
                            word_mention = doc[sent_id][word_offset:word_offset + len(tokens)]
                            parts = {"sent_id":sent_id,"char_start":i,"char_end":j, "entity_type":entity_type,
                                     "idx_span":(word_offset, word_offset + len(tokens)), "span":word_mention}
                            entity += [parts]
                        else:
                            print>> sys.stderr, "SUB SPAN ERROR", text, (i, j)
                            continue

                    # TODO: we assume continuous spans here
                    annotations[anno_id] = entity if not entity else entity[0]

                elif anno_id_prefix in [Brat.RELATION_ID,'*']:
                    anno_id, rela = row
                    rela_type, arg1, arg2 = rela.split()
                    arg1 = arg1.split(":")[1] if ":" in arg1 else arg1
                    arg2 = arg2.split(":")[1] if ":" in arg2 else arg2
                    annotations[anno_id] = (rela_type, arg1, arg2)

                elif anno_id_prefix == Brat.EVENT_ID:
                    print>> sys.stderr, "NotImplementedError: Events"
                    raise NotImplementedError

                elif anno_id_prefix == Brat.ATTRIB_ID:
                    print>> sys.stderr, "NotImplementedError: Attributes"

        return annotations

    def _parse_config(self, filename):
        """
        Parse BRAT
        :param filename:
        :return:
        """
        config = defaultdict(list)
        with open(filename, "rU") as fp:
            curr = None
            for line in fp:
                # skip comments
                line = line.strip()
                if not line or line[0] == '#':
                    continue
                # brat definition?
                m = re.search("^\[(.+)\]$", line)
                if m:
                    curr = m.group(1)
                    continue
                config[curr].append(line)

        # type-specific parsing
        tmp = []
        for item in config['relations']:
            m = re.search("^(.+)\s+Arg1:(.+),\s*Arg2:(.+),*\s*(.+)*$", item)
            name, arg1, arg2 = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            # convert relations to camel case
            name = self._get_normed_rela_name(name)
            arg2 = arg2.split(",")[0] # strip any <rel-type> defs
            arg1 = arg1.split("|")
            arg2 = arg2.split("|")
            tmp.append((name,arg1,arg2))
        config['relations'] = tmp

        tmp = []
        for item in config['attributes']:
            name, arg = item.split()
            arg = arg.split(":")[-1]
            tmp.append((name, arg))
        config['attributes'] = tmp

        return config

    def _create_candidate_subclasses(self, config):
        """
        Given a BRAT config file, create Snorkel candidate subclasses.
        NOTE: This method has a lot of hacks to deal with the schema definition limitations in Snorkel

        :param config:
        :return:
        """
        for class_name in config['entities']:
            try:
                # TODO: we strip nesting of entity defs, since Snorkel doesn't support hierarchical entity types
                class_name = class_name.strip()
                # see http://brat.nlplab.org/configuration.html#advanced-entities for advanced entity config
                # skip disabled types or seperators (these only display in the BRAT is-a hierarchy)
                if class_name[0] in ['!','-']:
                    continue
                self.subclasses[class_name] = candidate_subclass(class_name, [class_name.lower()])
                print('CREATED TYPE Entity({},[{}])'.format(class_name, class_name.lower()))
            except:
                pass

        # NOTE: relations must be uniquely named
        for item in config['relations']:
            name, arg1, arg2 = item
            #  Skip <ENTITY> tags; the generic entity argument (currently unsupported)
            ignore_args = {'<ENTITY>'}
            if ignore_args.intersection(arg1) or ignore_args.intersection(arg2):
                continue

            # TODO: Assume simple relation types *without* multiple argument types
            if (len(arg1) > 1 or len(arg2) > 1) and arg1 != arg2:
                print("Error: Snorkel currently does not support multiple argument types per relation", file=sys.stderr)

            try:
                args = sorted(set(arg1 + arg2))

                # fix for relations across the same type
                if len(arg1 + arg2) > 1 and len(set(arg1 + arg2)) == 1:
                    args = ["{}1".format(args[0]),"{}2".format(args[0])]

                args = map(lambda x:x.lower(),args)
                name = name.replace("-","_")

                self.subclasses[name] = candidate_subclass(name, args)
                print('CREATED TYPE Relation({},{})'.format(name, args))
            except Exception as e:
                print(e)


    def _create_config(self, candidate_types):
        """
        Export a minimal BRAT configuration schema defining
        a binary relation and two argument types.

        TODO: Model richer semantics here (asymmetry, n-arity relations)

        :param candidate_type:
        :return:
        """
        entity_defs, rela_defs = [], []
        for stype in candidate_types:
            rel_type = str(stype.type).rstrip(".type")
            arg_types = [key.rstrip("_id") for key in stype.__dict__ if "_id" in key]
            arg_types = [name[0].upper()+name[1:] for name in arg_types]
            entity_defs.extend(arg_types)
            if len(arg_types) > 1:
                rela_name = [str(stype.type).replace(".type","")] + arg_types
                rela_defs.append("{}\tArg1:{}, Arg2:{}".format(*rela_name))

        entity_defs = set(entity_defs)
        rela_defs = set(rela_defs)
        return self.brat_tmpl.format("\n".join(entity_defs), "\n".join(rela_defs), "", "")

    def _create_candidates(self, annotations, annotator_name, clear=True):
        """
        TODO: Add simpler candidate instantiation helper functions

        :return:
        """
        # create stable annotation labels
        stable_labels_by_type = defaultdict(list)

        for name in annotations:
            if annotations[name]:
                spans = [key for key in annotations[name] if key[0] == Brat.TEXT_BOUND_ID]
                relations = [key for key in annotations[name] if key[0] in [Brat.RELATION_ID]]

                # create span labels
                spans = {key:"{}::span:{}:{}".format(name, annotations[name][key]["char_start"],
                                                     annotations[name][key]["char_end"]) for key in spans}
                for key in spans:
                    entity_type = annotations[name][key]['entity_type']
                    stable_labels_by_type[entity_type].append(spans[key])

                # create relation labels
                for key in relations:
                    rela_type, arg1, arg2 = annotations[name][key]
                    rela = sorted([[annotations[name][arg1]["entity_type"], spans[arg1]],
                                    [annotations[name][arg2]["entity_type"],spans[arg2]]])
                    stable_labels_by_type[rela_type].append("~~".join(zip(*rela)[1]))

        # create stable labels
        # NOTE: we store each label class type in a different split so that it is compatible with
        # the current version of 'reload_annotator_labels', where we create candidates by split id
        for i, class_type in enumerate(stable_labels_by_type):

            for context_stable_id in stable_labels_by_type[class_type]:
                query = self.session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_id)
                query = query.filter(StableLabel.annotator_name == annotator_name)
                if query.count() != 0:
                    continue
                self.session.add(StableLabel(context_stable_ids=context_stable_id, split=i,
                                             annotator_name=annotator_name, value=1))

        abs_offsets = {}
        entity_types = defaultdict(list)
        for i, class_type in enumerate(stable_labels_by_type):

            if class_type in self.subclasses:
                class_name = self.subclasses[class_type]
            else:
                class_name = self.subclasses[self._get_normed_rela_name(class_type)]

            for et in stable_labels_by_type[class_type]:
                contexts = et.split('~~')
                spans = []

                for c,et in zip(contexts,class_name.__argnames__):
                    stable_id = c.split(":")
                    name, offsets = stable_id[0], stable_id[-2:]
                    span = list(map(int, offsets))

                    doc = self.session.query(Document).filter(Document.name == name).one()
                    if name not in abs_offsets:
                        abs_offsets[name] = abs_doc_offsets(doc)

                    for j,offset in enumerate(abs_offsets[name]):
                        if span[0] >= offset[0] and span[1] <= offset[1]:
                            try:
                                tc = TemporarySpan(char_start=span[0]-offset[0], char_end=span[1]-offset[0]-1,
                                                   sentence=doc.sentences[j])
                                tc.load_id_or_insert(self.session)
                                spans.append(tc)
                            except Exception as e:
                                print("BRAT candidate conversion error", len(doc.sentences), j)
                                print(e)

                entity_types[class_type].append(spans)

        for i, class_type in enumerate(stable_labels_by_type):

            if class_type in self.subclasses:
                class_name = self.subclasses[class_type]
            else:
                class_name = self.subclasses[self._get_normed_rela_name(class_type)]

            if clear:
                self.session.query(Candidate).filter(Candidate.split == i).delete()

            candidate_args = {'split': i}
            for args in entity_types[class_type]:
                for j, arg_name in enumerate(class_name.__argnames__):
                    candidate_args[arg_name + '_id'] = args[j].id

                candidate = class_name(**candidate_args)
                self.session.add(candidate)

        self.session.commit()


def _group_by_document(candidates):
    """

    :param candidates:
    :return:
    """
    doc_index = defaultdict(list)
    for c in candidates:
        name = c[0].sentence.document.name
        doc_index[name].append(c)
    return doc_index


def abs_doc_offsets(doc):
    """

    :param doc:
    :return:
    """
    abs_char_offsets = []
    for sent in doc.sentences:
        stable_id = sent.stable_id.split(":")
        name, offsets = stable_id[0], stable_id[-2:]
        offsets = map(int, offsets)
        abs_char_offsets.append(offsets)
    return abs_char_offsets


def doc_to_text(doc, sent_delim='\n'):
    """
    Convert document object to original text represention.
    Assumes parser offsets map to original document offsets
    :param doc:
    :param sent_delim:
    :return:
    """
    text = []
    for sent in doc.sentences:
        offsets = map(int, sent.stable_id.split(":")[-2:])
        char_start, char_end = offsets
        text.append({"text": sent.text, "char_start": char_start, "char_end": char_end})

    s = ""
    for i in range(len(text) - 1):
        gap = text[i + 1]['char_start'] - text[i]['char_end']
        s += text[i]['text'] + (sent_delim * gap)

    return s


class Brat(object):
    """
    Snorkel Import/Export for
    Brat Rapid Annotation Tool
    http://brat.nlplab.org/

    Brat uses standoff annotation format (see: http://brat.nlplab.org/standoff.html)

    Annotation ID Types
    T: text-bound annotation
    R: relation
    E: event
    A: attribute
    M: modification (alias for attribute, for backward compatibility)
    N: normalization [new in v1.3]
    #: note

    Caveats:
    (1) Attributes, normalization, and notes are added as candidate meta information

    """

    TEXT_BOUND_ID = 'T'
    RELATION_ID = 'R'
    EVENT_ID = 'E'
    ATTRIB_ID = 'A'
    MOD_ID = 'M'
    NORM_ID = 'N'
    NOTE_ID = '#'

    def __init__(self, session, tmpl_path='tmpl.config', encoding="utf-8", verbose=True):
        """

        :param session:
        :param tmpl_path:
        :param encoding:
        :param verbose:
        """
        self.session = session
        self.encoding = encoding
        self.verbose = verbose

        # load brat config template
        mod_path = "{}/{}".format(os.path.abspath(os.path.dirname(__file__)), tmpl_path)
        self.brat_tmpl = "".join(open(mod_path, "rU").readlines())

        # snorkel dynamic types
        self.subclasses = {}

    def import_project(self, input_dir, annotations_only=True, annotator_name='brat', num_threads=1, parser=None):
        """

        :param input_dir:
        :param autoreload:
        :param num_threads:
        :param parser:
        :return:
        """
        config_path = "{}/{}".format(input_dir, "annotation.conf")
        if not os.path.exists(config_path):
            print("Fatal error: missing 'annotation.conf' file", file=sys.stderr)
            return

        # load brat config (this defines relation and argument types)
        config = self._load_config(config_path)
        anno_filelist = set([os.path.basename(fn).strip(".ann") for fn in glob.glob(input_dir + "/*.ann")])

        # import standoff annotations for all documents
        annotations = {}
        for fn in anno_filelist:
            txt_fn = "{}/{}.txt".format(input_dir, fn)
            ann_fn = "{}/{}.ann".format(input_dir, fn)
            if os.path.exists(txt_fn) and os.path.exists(ann_fn):
                annotations[fn] = self._parse_annotations(txt_fn, ann_fn)

        # by default, we parse and import all project documents
        if not annotations_only:
            self._parse_documents(input_dir + "/*.txt", num_threads, parser)

        # create types
        self._create_candidate_subclasses(config)

        # create candidates
        self._create_candidates(annotations, annotator_name)

    def export_by_candidate_marginals(self, output_dir, positive_only_labels=True):
        """
        :param output_dir:
        :positive_only_labels
        :return:
        """
        print(f"Writing to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        candidates = self.session.query(Candidate).filter(Candidate.split == 0).all()
        print('Grouping candidates by document')
        doc_index = _group_candidates_by_document(candidates)
        snorkel_types = {type(c) for c in candidates}
        configuration_string = self._create_config_from_candidate_types(snorkel_types)

        with open(os.path.join(output_dir, 'annotation.conf'), 'w') as conf_file:
            conf_file.write(configuration_string)

        # iterate over the documents
        for name in tqdm(doc_index):
            # write the text
            with open(os.path.join(output_dir, f'{name}.txt'), 'w') as text_file:
                text = "\n".join([sentence.text for sentence in doc_index[name][0][0].sentence.document.sentences])
                text_file.write(text)

            # write the annotation file
            with open(os.path.join(output_dir, f'{name}.ann'), 'w') as ann_file:
                annotation_tuples = []
                for c in doc_index[name]:
                    if positive_only_labels and c.training_marginal <= 0.5:
                        continue
                    sentence_start = sum(
                        len(sentence.text) for sentence in c[0].sentence.document.sentences[:c[0].sentence.position])
                    char_start = sentence_start + c[0].char_start
                    char_end = sentence_start + c[0].char_end + 1
                    text = c[0].get_span()
                    annotation_tuples.append((c.__class__.__name__, char_start, char_end, text))

                annotation_tuples.sort(key=lambda tuple: tuple[1])
                lines = [
                    f'T{i + 1}\t{annotation_tuple[0]} {annotation_tuple[1]} {annotation_tuple[2]}\t{annotation_tuple[3]}\n'
                    for i, annotation_tuple in enumerate(annotation_tuples)]
                ann_file.writelines(lines)

    def export_by_noisy_tagged_sentences(self, output_dir, num_sentences, entity_names):
        """
        :param output_dir:
        :return:
        """
        print(f"Writing to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        noisy_tagged_sentences = self.session.query(NoisyTaggedSentence).all()
        print('Grouping candidates by document')
        doc_index = _group_noisy_sentences_by_document(noisy_tagged_sentences, num_sentences)
        candidates = self.session.query(Candidate).filter(Candidate.type.in_({e.lower() for e in entity_names})).all()
        type_candidate_ids = {c.id for c in candidates}
        snorkel_types = {type(c) for c in candidates}
        configuration_string = self._create_config_from_candidate_types(snorkel_types)

        with open(os.path.join(output_dir, 'annotation.conf'), 'w') as conf_file:
            conf_file.write(configuration_string)

        # iterate over the documents
        for name in tqdm(doc_index):
            for i, noisy_tagged_sentences in enumerate(doc_index[name]):
                # write the text
                if i == 0:
                    with open(os.path.join(output_dir, f'{name}_{i}.txt'), 'w') as text_file:
                        text = " ".join(
                            [sentence.text for sentence in doc_index[name][0][0].sentence.document.sentences])
                        text_file.write(text)
                else:
                    os.symlink(os.path.join(output_dir, f'{name}_{0}.txt'), os.path.join(output_dir, f'{name}_{i}.txt'))
                # write the annotation file
                with open(os.path.join(output_dir, f'{name}_{i}.ann'), 'w') as ann_file:
                    candidate_ids = set(
                        itertools.chain.from_iterable(
                            [[candidate_id for candidate_id in s.candidate_ids if candidate_id in type_candidate_ids]
                             for s in noisy_tagged_sentences]))
                    annotation_tuples = []
                    if candidate_ids:
                        candidates = self.session.query(Candidate).filter(Candidate.id.in_(candidate_ids)).all()
                        for c in candidates:
                            sentence_start = sum(
                                len(sentence.text) for sentence in
                                c[0].sentence.document.sentences[:c[0].sentence.position])
                            char_start = sentence_start + c[0].char_start
                            char_end = sentence_start + c[0].char_end + 1
                            text = c[0].get_span()
                            annotation_tuples.append((c.__class__.__name__, char_start, char_end, text))

                    annotation_tuples.sort(key=lambda tuple: tuple[1])
                    lines = [
                        f'T{i + 1}\t{annotation_tuple[0]} {annotation_tuple[1]} {annotation_tuple[2]}\t{annotation_tuple[3]}\n'
                        for i, annotation_tuple in enumerate(annotation_tuples)]
                    ann_file.writelines(lines)

    def _parse_documents(self, input_path, num_threads, parser):
        """

        :param input_path:
        :param num_threads:
        :param parser:
        :return:
        """
        doc_preprocessor = TextDocPreprocessor(path=input_path, encoding=self.encoding)
        corpus_parser = CorpusParser(parser)
        corpus_parser.apply(doc_preprocessor, parallelism=num_threads)

    def _parse_annotations(self, txt_filename, ann_filename):
        """
        Use parser to import BRAT backoff format
        TODO: Currently only supports Entities & Relations

        :param txt_filename:
        :param ann_filename:
        :return:
        """
        annotations = {}

        # load document
        doc = []
        with codecs.open(txt_filename, "rU", encoding=self.encoding) as fp:
            for line in fp:
                doc += [line.strip().split()]

        # build doc string and char to word index
        doc_str = ""
        char_idx = {}
        for i, sent in enumerate(doc):
            for j in range(0, len(sent)):
                char_idx[len(doc_str)] = (i, j)
                for ch in sent[j]:
                    doc_str += ch
                    char_idx[len(doc_str)] = (i, j)
                doc_str += " " if j != len(sent) - 1 else "\n"
        doc_str = doc_str.strip()

        # load annotations
        with codecs.open(ann_filename, "rU", encoding=self.encoding) as fp:
            for line in fp:
                row = line.strip().split("\t")
                anno_id_prefix = row[0][0]

                if anno_id_prefix == Brat.TEXT_BOUND_ID:
                    anno_id, entity, text = row
                    entity_type = entity.split()[0]
                    spans = [list(map(int, x.split())) for x in entity.lstrip(entity_type).split(";")]

                    # discontinuous mentions
                    if len(spans) != 1:
                        print("NotImplementedError: Discontinuous Spans", file=sys.stderr)
                        continue

                    entity = []
                    for (i, j) in spans:
                        if i in char_idx:
                            mention = doc_str[i:j]
                            tokens = mention.split()
                            sent_id, word_offset = char_idx[i]
                            word_mention = doc[sent_id][word_offset:word_offset + len(tokens)]
                            parts = {"sent_id": sent_id, "char_start": i, "char_end": j, "entity_type": entity_type,
                                     "idx_span": (word_offset, word_offset + len(tokens)), "span": word_mention}
                            entity += [parts]
                        else:
                            print("SUB SPAN ERROR", text, (i, j), file=sys.stderr)
                            continue

                    # TODO: we assume continuous spans here
                    annotations[anno_id] = entity if not entity else entity[0]

                elif anno_id_prefix == Brat.RELATION_ID:
                    anno_id, rela = row
                    rela_type, arg1, arg2 = rela.split()

                    arg1 = arg1.split(":")[1] if ":" in arg1 else arg1
                    arg2 = arg2.split(":")[1] if ":" in arg2 else arg2

                    annotations[anno_id] = (rela_type, arg1, arg2)

                elif anno_id_prefix == Brat.EVENT_ID:
                    print("NotImplementedError: Events", file=sys.stderr)
                    raise NotImplementedError

                elif anno_id_prefix == Brat.ATTRIB_ID:
                    print("NotImplementedError: Attributes", file=sys.stderr)

        return annotations

    def _create_candidate_subclasses(self, config):
        """
        Given a BRAT config file, create Snorkel candidate subclasses.

        :param config:
        :return:
        """
        for class_name in config['entities']:
            try:
                self.subclasses[class_name] = candidate_subclass(class_name, [class_name.lower()])
                print(('Entity({},[{}])'.format(class_name, class_name.lower())))
            except:
                pass

        for item in config['relations']:
            m = re.search("^(.+)\s+Arg1:(.+),\s*Arg2:(.+),*\s*(.+)*$", item)
            name, arg1, arg2 = m.group(1), m.group(2), m.group(3)

            arg1 = arg1.lower().split("|")
            arg2 = arg2.lower().split("|")

            # TODO: Assume simple relation types *without* multiple argument types
            if (len(arg1) > 1 or len(arg2) > 1) and arg1 != arg2:
                print("Error: Snorkel does not support multiple argument types", file=sys.stderr)

            try:
                args = sorted(set(arg1 + arg2))
                self.subclasses[name] = candidate_subclass(name, args)
                print('Relation({},{})'.format(name, args))
            except:
                pass

    def _load_config(self, filename):
        """

        :param filename:
        :return:
        """
        config = defaultdict(list)
        with open(filename, "rU") as fp:
            curr = None
            for line in fp:
                # skip comments
                line = line.strip()
                if not line or line[0] == '#':
                    continue
                # brat definition?
                m = re.search("^\[(.+)\]$", line)
                if m:
                    curr = m.group(1)
                    continue
                config[curr].append(line)

        return config

    def _create_config(self, candidate_type):
        """
        Export a minimal BRAT configuration schema defining
        a binary relation and two argument types.

        TODO: Model richer semantics here (asymmetry, n-arity relations)

        :param candidate_type:
        :return:
        """
        rel_type = str(candidate_type.type).rstrip(".type")
        arg_types = [key.rstrip("_cid") for key in candidate_type.__dict__ if "_cid" in key]

        entity_defs = "\n".join(arg_types)
        rela_def = "Arg1:{}, Arg2:{}".format(*arg_types) if len(arg_types) == 2 else ""
        return self.brat_tmpl.format(entity_defs, rela_def, "", "")

    def _create_config_from_candidate_types(self, candidate_types):
        """
        Export a minimal BRAT configuration schema defining
        a multiple argument types        

        :param candidate_types:
        :return:
        """
        if type(list(candidate_types)[0]) != str:
            candidate_types = [candidate_type.__name__ for candidate_type in candidate_types]

        entity_defs = "\n".join(candidate_types)
        return self.brat_tmpl.format(entity_defs, "", "", "")

    def _create_candidates(self, annotations, annotator_name, clear=True):
        """

        :return:
        """
        # create stable annotation labels
        stable_labels_by_type = defaultdict(list)

        for name in annotations:
            if annotations[name]:
                spans = [key for key in annotations[name] if key[0] == Brat.TEXT_BOUND_ID]
                relations = [key for key in annotations[name] if key[0] == Brat.RELATION_ID]

                # create span labels
                spans = {key: "{}::span:{}:{}".format(name, annotations[name][key]["char_start"],
                                                      annotations[name][key]["char_end"]) for key in spans}
                for key in spans:
                    entity_type = annotations[name][key]['entity_type']
                    stable_labels_by_type[entity_type].append(spans[key])

                # create relation labels
                for key in relations:
                    rela_type, arg1, arg2 = annotations[name][key]
                    rela = sorted([[annotations[name][arg1]["entity_type"], spans[arg1]],
                                   [annotations[name][arg2]["entity_type"], spans[arg2]]])
                    stable_labels_by_type[rela_type].append("~~".join(zip(*rela)[1]))

        # create stable labels
        # NOTE: we store each label class type in a different split so that it is compatible with
        # the current version of 'reload_annotator_labels', where we create candidates by split id
        for i, class_type in enumerate(stable_labels_by_type):

            for context_stable_id in stable_labels_by_type[class_type]:
                query = self.session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_id)
                query = query.filter(StableLabel.annotator_name == annotator_name)
                if query.count() != 0:
                    continue
                self.session.add(StableLabel(context_stable_ids=context_stable_id, split=i,
                                             annotator_name=annotator_name, value=1))

        abs_offsets = {}
        entity_types = defaultdict(list)
        for i, class_type in enumerate(stable_labels_by_type):

            class_name = self.subclasses[class_type]
            for et in stable_labels_by_type[class_type]:
                contexts = et.split('~~')
                spans = []
                for c, et in zip(contexts, class_name.__argnames__):
                    stable_id = c.split(":")
                    name, offsets = stable_id[0], stable_id[-2:]
                    span = list(map(int, offsets))
                    if name not in abs_offsets:
                        doc = self.session.query(Document).filter(Document.name == name).one()
                        abs_offsets[name] = abs_doc_offsets(doc)

                    for j, offset in enumerate(abs_offsets[name]):
                        if span[0] >= offset[0] and span[1] <= offset[1]:
                            tc = TemporarySpan(char_start=span[0] - offset[0], char_end=span[1] - offset[0] - 1,
                                               sentence=doc.sentences[j])
                            tc.load_id_or_insert(self.session)
                            spans.append(tc)

                entity_types[class_type].append(spans)

        for i, class_type in enumerate(stable_labels_by_type):

            if clear:
                self.session.query(Candidate).filter(Candidate.split == i).delete()

            candidate_args = {'split': i}
            for args in entity_types[class_type]:
                for j, arg_name in enumerate(self.subclasses[class_type].__argnames__):
                    candidate_args[arg_name + '_id'] = args[j].id

                candidate = self.subclasses[class_type](**candidate_args)
                self.session.add(candidate)

        self.session.commit()


def _group_candidates_by_document(candidates):
    """

    :param candidates:
    :return:
    """
    doc_index = defaultdict(list)
    for c in candidates:
        name = c[0].sentence.document.name
        doc_index[name].append(c)
    return doc_index


def _group_noisy_sentences_by_document(noisy_tagged_sentences, num_sentences):
    doc_index = defaultdict(list)
    for t in noisy_tagged_sentences:
        name = t.sentence.document.name
        doc_index[name].append(t)
    for name, noisy_tagged_sentences in doc_index.items():
        noisy_tagged_sentences.sort(key=lambda s: s.sentence.position)
        doc_index[name] = [[noisy_tagged_sentences[i + j * num_sentences]
                            for j
                            in range(int(len(noisy_tagged_sentences) / num_sentences))]
                           for i
                           in range(num_sentences)]
    return doc_index


def abs_doc_offsets(doc):
    """

    :param doc:
    :return:
    """
    abs_char_offsets = []
    for sent in doc.sentences:
        stable_id = sent.stable_id.split(":")
        name, offsets = stable_id[0], stable_id[-2:]
        offsets = list(map(int, offsets))
        abs_char_offsets.append(offsets)
    return abs_char_offsets
