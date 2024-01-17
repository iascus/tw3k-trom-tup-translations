import difflib
import logging
from abc import abstractmethod

import numpy as np
import pandas as pd
import pandas.testing as pdt

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Step:
    step_no = None
    data = None

    def __repr__(self):
        return f'<Step {self.name} - {self.data.shape if self.data is not None else 0}>'

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def filename(self):
        return f'{self.step_no:02}.{self.name}.csv'

    @property
    def out_filepath(self):
        return f'csv/out/{self.filename}'

    @property
    def xls_filepath(self):
        return f'csv/xls/{self.filename}'

    @property
    def diff_filepath(self):
        return f'csv/diff/{self.filename}.diff'

    def save(self):
        self.data.to_csv(self.out_filepath, index=False)
        logger.info(f'Saved to {self.out_filepath}')

    @abstractmethod
    def _run(self):
        raise NotImplementedError()

    def run(self):
        logger.info(f'Running {self.name}')
        self.data = self._run()

    def load_from_xls(self):
        logger.info(f'Loading {self.xls_filepath}')
        return pd.read_csv(self.xls_filepath)

    def compare_with_xls(self):
        logger.info('Comparing with xls data')
        with (
            open(self.out_filepath, encoding='utf8') as out_file,
            open(self.xls_filepath, encoding='utf8') as xls_file,
            open(self.diff_filepath, encoding='utf8', mode='w') as diff_file,
        ):
            diff = difflib.unified_diff(
                out_file.read().replace(',TRUE', ',True').replace(',FALSE', ',False').splitlines(keepends=True),
                (xls_file.read().rstrip() + '\n').replace(',TRUE', ',True').replace(',FALSE', ',False').splitlines(keepends=True),
                self.out_filepath, self.xls_filepath,
                n=0,
            )
            diff_file.writelines(diff)


class InputFile(Step):

    def _run(self):
        return self.load_from_xls()


class VanillaZhcn(InputFile):
    step_no = 1


class VanillaZhtw(InputFile):
    step_no = 2


class VanillaEng(InputFile):
    step_no = 3


class TromZhcn(InputFile):
    step_no = 4


class TromZhtw(InputFile):
    step_no = 5


class TromEng(InputFile):
    step_no = 6


class KeyMapOverride(InputFile):
    step_no = 7


class EngToVanillaKey(Step):
    step_no = 8

    def __init__(self, trom_eng, key_map_override, vanilla_eng, vanilla_zhtw, vanilla_zhcn):
        self.trom_eng = trom_eng
        self.key_map_override = key_map_override
        self.vanilla_eng = vanilla_eng
        self.vanilla_zhtw = vanilla_zhtw
        self.vanilla_zhcn = vanilla_zhcn

    def _run(self):
        trom_eng = self.trom_eng.data.dropna(subset='Text').set_index('Text')
        vanilla_eng = self.vanilla_eng.data.drop_duplicates(subset=['Text']).set_index('Text')[['Key']]
        vanilla_eng = vanilla_eng.rename({'Key': 'VanillaKey'}, axis=1)
        vanilla_zhcn = self.vanilla_zhcn.data.set_index('Key')[['Text']].rename({'Text': 'zh-cn'}, axis=1)
        vanilla_zhtw = self.vanilla_zhtw.data.set_index('Key')[['Text']].rename({'Text': 'zh-tw'}, axis=1)
        data = trom_eng.join(vanilla_eng, how='left').reset_index().set_index('Key')
        data = data.join(
            self.key_map_override.data.set_index('TromKey').rename({'VanillaKey': 'VanillaKeyOverride'}, axis=1)
        ).reset_index()
        data.index = np.where(~pd.isna(data['VanillaKeyOverride']), data['VanillaKeyOverride'], data['VanillaKey'])
        data = data.join(vanilla_zhcn).join(vanilla_zhtw).reset_index()[
            ['Key', 'Text', 'Tooltip', 'VanillaKey', 'VanillaKeyOverride', 'zh-tw', 'zh-cn', 'File']]
        return data


class LookupByUnit(InputFile):
    step_no = 9


class LookupByText(InputFile):
    step_no = 10


class LookupByKey(InputFile):
    step_no = 11


class MapByTextZhcn(InputFile):
    step_no = 12


class MapByTextZhtw(InputFile):
    step_no = 13


class MapByKeyZhcn(Step):
    step_no = 14
    lang_col = 'zh-cn'

    def __init__(self, trom_eng, eng_to_vanilla_key, trom_zh, lookup_by_key, map_by_text_zh):
        self.trom_eng = trom_eng
        self.eng_to_vanilla_key = eng_to_vanilla_key
        self.trom_zh = trom_zh
        self.lookup_by_key = lookup_by_key
        self.map_by_text_zh = map_by_text_zh

    def _run(self):
        data = self.trom_eng.data.fillna('').rename({'Text': 'English'}, axis=1)
        vanilla = self.eng_to_vanilla_key.data.drop_duplicates(subset=['Key'])
        vanilla = vanilla[['Key', self.lang_col]].rename({self.lang_col: 'Vanilla'}, axis=1)
        trom_zh = self.trom_zh.data.drop_duplicates(subset=['Key'])[['Key', 'Text']]
        lookup_by_key = self.lookup_by_key.data[['Key', self.lang_col]].rename({self.lang_col: 'Override by key'}, axis=1)
        data = data.merge(lookup_by_key, on='Key', how='left')
        data = data.merge(vanilla, on='Key', how='left')
        data = data.merge(trom_zh.rename({'Text': 'Exact'}, axis=1), on='Key', how='left')
        data = data.merge(trom_zh.rename({'Text': 'Eng-short', 'Key': 'Short Key'}, axis=1), on='Short Key', how='left')

        data['Text'] = data['English']
        data['Source'] = 'Missing'
        map_by_text_zh = self.map_by_text_zh.data[['Text', 'Mapped']].drop_duplicates(subset=['Text']).rename({'Mapped': 'Mapped by text'}, axis=1)
        data = data.merge(map_by_text_zh, on='Text', how='left')
        for col in [
            'Eng-short',
            'Exact',
            'Vanilla',
            'Mapped by text',
            'Override by key',
        ]:
            data.loc[~pd.isna(data[col]), ['Text']] = data[col]
            data.loc[~pd.isna(data[col]), ['Source']] = col
        data['File'] = data['File'].str.replace('text/_hvo/', '')
        data = data.reset_index()[[
            'Key', 'Text', 'Tooltip', 'English', 'Override by key',
            'Mapped by text', 'Vanilla', 'Exact', 'Eng-short',
            'Source', 'Short Key', 'File',
        ]]
        return data


class MapByKeyZhtw(MapByKeyZhcn):
    step_no = 15
    lang_col = 'zh-tw'


class FinalZhcn(Step):
    step_no = 16

    def __init__(self, map_by_key_zh):
        self.map_by_key_zh = map_by_key_zh

    def _run(self):
        data = self.map_by_key_zh.data[['Key', 'Text', 'Tooltip']]
        return data


class FinalZhtw(FinalZhcn):
    step_no = 17


def main():
    vanilla_zhcn = VanillaZhcn()
    vanilla_zhtw = VanillaZhtw()
    vanilla_eng = VanillaEng()
    trom_zhcn = TromZhcn()
    trom_zhtw = TromZhtw()
    trom_eng = TromEng()
    key_map_override = KeyMapOverride()
    eng_to_vanilla_key = EngToVanillaKey(trom_eng, key_map_override, vanilla_eng, vanilla_zhtw, vanilla_zhcn)
    lookup_by_unit = LookupByUnit()
    lookup_by_text = LookupByText()
    lookup_by_key = LookupByKey()
    map_by_text_zhcn = MapByTextZhcn()
    map_by_text_zhtw = MapByTextZhtw()
    map_by_key_zhcn = MapByKeyZhcn(trom_eng, eng_to_vanilla_key, trom_zhcn, lookup_by_key, map_by_text_zhcn)
    map_by_key_zhtw = MapByKeyZhtw(trom_eng, eng_to_vanilla_key, trom_zhtw, lookup_by_key, map_by_text_zhtw)
    final_zhcn = FinalZhcn(map_by_key_zhcn)
    final_zhtw = FinalZhtw(map_by_key_zhtw)

    for step in [
        vanilla_zhcn,
        vanilla_zhtw,
        vanilla_eng,
        trom_zhcn,
        trom_zhtw,
        trom_eng,
        key_map_override,
        eng_to_vanilla_key,
        lookup_by_unit,
        lookup_by_text,
        lookup_by_key,
        map_by_text_zhcn,
        map_by_text_zhtw,
        map_by_key_zhcn,
        map_by_key_zhtw,
        final_zhcn,
        final_zhtw,
    ]:
        step.run()
        step.save()
        step.compare_with_xls()


if __name__ == '__main__':
    with pd.option_context(
        'display.width', None,
        'display.max_columns', None,
    ):
        main()
