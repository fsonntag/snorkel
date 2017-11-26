from sqlalchemy import (
    Column, String, Integer, Float, Boolean, ForeignKey, UniqueConstraint,
    MetaData
)

from .meta import SnorkelBase

class Attribute(SnorkelBase):
    """
    This table stores Context object attribute pairs. This is useful for storing non-materialized
    attribute metadata (e.g., document-level keyword tags, publication source) that could be useful
    for labeling function design.
    """
    __tablename__ = 'attribute'
    id         = Column(Integer, primary_key=True)
    context_id = Column(Integer, ForeignKey('context.id'))
    name       = Column(String, nullable=False)
    value      = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint(context_id, name, value),
    )

    def __repr__(self):
        return "<Attribute: {}({}:{})>".format(self.context_id, self.name, self.value)




