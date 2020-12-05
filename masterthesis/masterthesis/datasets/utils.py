import re


class LineReader(object):

    def __init__(self, annotation_str, separator):
        columns = re.split(separator, annotation_str)
        self.columns = list(map(lambda s: s.strip(), columns))
        self.offset = 0

    def __call__(self, count: int = 1, func=None):
        assert count > 0

        _func = func if func else lambda x: x

        if count == 1:
            record = _func(self.columns[self.offset])
        else:
            record = []
            for i in range(self.offset, self.offset + count):
                record.append(_func(self.columns[i]))

        self.skip(count)
        return record

    def __len__(self):
        return len(self.columns) - self.offset

    def __bool__(self):
        return len(self.columns) > self.offset

    def skip(self, count):
        self.offset += count

    def has_columns(self, count):
        return len(self) >= count
