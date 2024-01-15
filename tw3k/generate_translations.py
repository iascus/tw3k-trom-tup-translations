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

    def save(self):
        filename = f'csv/out/{self.filename}'
        self.data.to_csv(filename, index=False)
        logger.info(f'Saved to {filename}')

    @abstractmethod
    def _run(self):
        raise NotImplementedError()

    def run(self):
        logger.info(f'Running {self.name}')
        self.data = self._run()

    def load_from_xls(self):
        filename = f'csv/xls/{self.filename}'
        logger.info(f'Loading {filename}')
        return pd.read_csv(filename)

    def compare_with_xls(self):
        return
        xls_data = self.load_from_xls()
        logger.info('Comparing with xls data')
        pdt.assert_frame_equal(self.data, xls_data)


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


class EngToVanillaKey(InputFile):
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
        data = data.join(self.key_map_override.data.set_index('TromKey').rename({'VanillaKey': 'VanillaKeyOverride'}, axis=1)).reset_index()
        data.index = np.where(pd.isna(data['VanillaKeyOverride']), data['VanillaKey'], data['VanillaKeyOverride'])
        data = data.join(vanilla_zhcn).join(vanilla_zhtw).reset_index()[
            ['Key', 'Text', 'Tooltip', 'VanillaKey', 'VanillaKeyOverride', 'zh-tw', 'zh-cn', 'File']]
        return data


def main():
    vanilla_zhcn = VanillaZhcn()
    vanilla_zhtw = VanillaZhtw()
    vanilla_eng = VanillaEng()
    trom_zhcn = TromZhcn()
    trom_zhtw = TromZhtw()
    trom_eng = TromEng()
    key_map_override = KeyMapOverride()
    eng_to_vanilla_key = EngToVanillaKey(trom_eng, key_map_override, vanilla_eng, vanilla_zhtw, vanilla_zhcn)

    for step in [
        vanilla_zhcn,
        vanilla_zhtw,
        vanilla_eng,
        trom_zhcn,
        trom_zhtw,
        trom_eng,
        key_map_override,
        eng_to_vanilla_key,
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
