import logging
from abc import abstractmethod

import pandas as pd
import pandas.testing as pdt

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Step:
    step_no = None
    data = None

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def filename(self):
        return f'{self.step_no:02}.{self.name}.csv'

    def save(self):
        filename = f'csv/out/{self.filename}'
        self.data.to_csv(filename)
        logger.info(f'Saved to {filename}')

    @abstractmethod
    def _run(self):
        raise NotImplementedError()

    def run(self):
        logger.info(f'Running {self.name}')
        self._run()

    def load_from_xls(self):
        filename = f'csv/xls/{self.filename}'
        logger.info(f'Loading {filename}')
        return pd.read_csv(filename)

    def compare_with_xls(self):
        xls_data = self.load_from_xls()
        logger.info('Comparing with xls data')
        pdt.assert_frame_equal(self.data, xls_data)


class InputFile(Step):

    def _run(self):
        self.data = self.load_from_xls()


class VanillaZhcn(InputFile):
    step_no = 1


class VanillaZhtw(InputFile):
    step_no = 2


class VanillaEng(InputFile):
    step_no = 3


class TromZhcn(InputFile):
    step_no = 6


class TromZhtw(InputFile):
    step_no = 7


class TromEng(InputFile):
    step_no = 8


class KeyMapOverride(InputFile):
    step_no = 5


class EngToVanillaKey(Step):
    step_no = 4

    def __init__(self, trom_eng, vanilla_eng, vanilla_zhtw, vanilla_zhcn):
        self.trom_eng = trom_eng.data
        self.vanilla_eng = vanilla_eng.data
        self.vanilla_zhtw = vanilla_zhtw.data
        self.vanilla_zhcn = vanilla_zhcn.data

    def _run(self):
        print(self.data)


def main():
    vanilla_zhcn = VanillaZhcn()
    vanilla_zhtw = VanillaZhtw()
    vanilla_eng = VanillaEng()
    key_map_override = KeyMapOverride()
    trom_zhcn = TromZhcn()
    trom_zhtw = TromZhtw()
    trom_eng = TromEng()
    eng_to_vanilla_key = EngToVanillaKey(trom_eng, vanilla_eng, vanilla_zhtw, vanilla_zhcn)

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
    main()
