import glob
import logging
import os
import re
from abc import abstractmethod

import numpy as np
import pandas as pd

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
    def in_filepath(self):
        return f'csv/in/{self.name}.csv'

    @property
    def lookup_filepath(self):
        return f'csv/lookup/{self.name}.csv'

    @property
    def out_filepath(self):
        return f'csv/out/{self.filename}'

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

    def load_file(self, file_type):
        if file_type == 'in':
            path = self.in_filepath
        elif file_type == 'lookup':
            path = self.lookup_filepath
        else:
            raise ValueError(f'Unknown type: {file_type}')
        logger.info(f'Loading {path}')
        return pd.read_csv(path)


class InputFile(Step):
    def _run(self):
        return self.load_file('in')


class LookupFile(Step):
    def _run(self):
        return self.load_file('lookup')


class TsvFiles(Step):
    def _run(self):
        dfs = []
        for filepath in glob.glob(os.path.join('csv', 'in', self.dir_path, '**', '*.tsv'), recursive=True):
            if 'mtu_text' in filepath:
                continue
            with open(filepath, encoding='utf-8') as tsv_file:
                df = pd.read_csv(tsv_file, sep='\t', skiprows=[1])
                df.columns = ['Key', 'Text', 'Tooltip']
                df['File'] = os.path.basename(filepath)
                dfs.append(df)
        data = pd.concat(dfs).dropna(subset='Key').drop_duplicates(subset='Key').sort_values('Key')
        return data


class LocOutput(Step):

    tsv_filename = '!@hv_TEXT.tsv'

    def save(self):
        super().save()
        loc_filepath = f'csv/out/{self.lang_col}/{self.tsv_filename}'
        data = self.data.loc[self.data['Key'] != ''].dropna()
        data['Tooltip'] = data['Tooltip'].astype(str).str.lower()
        with open(loc_filepath, 'w', encoding='utf-8', newline='\n') as out_file:
            out_file.writelines([
                '\t'.join(str(col).lower() for col in self.data.columns) + '\n',
                f'#Loc;1;text/{self.tsv_filename}\n',
            ])
            data.to_csv(out_file, index=False, header=False, sep='\t', lineterminator='\n')
        logger.info(f'Saved to {loc_filepath}')


class VanillaZhcn(InputFile):
    step_no = 1


class VanillaZhtw(InputFile):
    step_no = 2


class VanillaEng(InputFile):
    step_no = 3


class TromZhcn(TsvFiles):
    step_no = 4
    dir_path = 'PikaManZhcn'


class TromZhtw(TsvFiles):
    step_no = 5
    dir_path = 'PikaManZhtw'


class TromEng(TsvFiles):
    step_no = 6
    dir_path = 'Trom3.9e'


class ProcrastinatorZhcn(TsvFiles):
    step_no = 7
    dir_path = 'ProcrastinatorZhcn'


class TromVanillaKeyOverride(LookupFile):
    step_no = 11


class LookupByText(LookupFile):
    step_no = 12


class LookupByKey(LookupFile):
    step_no = 13


class LookupByPattern(LookupFile):
    step_no = 14


class LookupByUnitName(LookupFile):
    step_no = 15


class LookupByUnitType(LookupFile):
    step_no = 16


class LookupByTextFragment(LookupFile):
    step_no = 17


class VanillaTranslations(Step):
    step_no = 21

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


class MapByPatternZhcn(Step):
    step_no = 22
    lang_col = 'zh-cn'

    def __init__(self, trom_eng, vanilla_translations, lookup_by_text, lookup_by_pattern, lookup_by_unit_name, lookup_by_unit_type, lookup_by_text_fragment):
        self.trom_eng = trom_eng
        self.vanilla_translations = vanilla_translations
        self.lookup_by_unit_name = lookup_by_unit_name
        self.lookup_by_unit_type = lookup_by_unit_type
        self.lookup_by_text = lookup_by_text
        self.lookup_by_pattern = lookup_by_pattern
        self.lookup_by_text_fragment = lookup_by_text_fragment

    @staticmethod
    def _lookup(found, matched, key, lookup):
        if key in matched:
            if matched[key] in lookup:
                found[key] = lookup[matched[key]]
            elif matched[key].startswith('{{') and matched[key].endswith('}}'):
                found[key] = matched[key]
        return found

    def _run(self):
        data = self.trom_eng.data.fillna('').copy()
        data = data.groupby(['Text']).agg(Tooltip=('Tooltip', 'first'), File=('File', 'first'), Count=('Key', 'count'), Key=('Key', 'first'))
        lookup_by_text = self.lookup_by_text.data[~pd.isna(self.lookup_by_text.data[self.lang_col])]
        lookup_by_text = lookup_by_text.set_index('eng')[[self.lang_col]].rename({self.lang_col: 'Mapped by text'}, axis=1)
        data = data.merge(lookup_by_text, left_index=True, right_index=True, how='left')
        data['Mapped by pattern'] = pd.Series(pd.NA, dtype='string')

        lookup_by_text_fragment = self.lookup_by_text_fragment.data.set_index('eng')[self.lang_col].to_dict()
        lookup_by_pattern = self.lookup_by_pattern.data.set_index(['KeyPattern', 'TextPattern'])[[self.lang_col]]
        lookup_by_unit_type = self.lookup_by_unit_type.data.set_index('Text')[self.lang_col].to_dict()
        lookup_by_unit_name = self.lookup_by_unit_name.data.set_index('Text')[self.lang_col].to_dict()
        lookup_by_text_vanilla = self.vanilla_translations.data[['Text', self.lang_col]].dropna().set_index('Text')[self.lang_col].to_dict()
        lookup_by_text_vanilla.update(lookup_by_text.to_dict()['Mapped by text'])
        lookup_by_text = lookup_by_text_vanilla

        compiled_regex = {}

        for text, row in data.iterrows():
            for idx, replacement in lookup_by_pattern.iterrows():
                key_pattern, text_pattern = idx
                if pd.isna(key_pattern):
                    key_pattern = None
                else:
                    key_pattern = compiled_regex.setdefault(key_pattern, re.compile(key_pattern))
                text_pattern = compiled_regex.setdefault(text_pattern, re.compile(text_pattern))
                replacement = replacement.iloc[0]
                if key_pattern and not re.match(key_pattern, row['Key']):
                    continue
                if matched := re.match(text_pattern, text):
                    matched = matched.groupdict()
                    found = {'repl': matched['repl']} if 'repl' in matched else {}
                    found = self._lookup(found, matched, 'text', lookup_by_text)
                    found = self._lookup(found, matched, 'text_fragment', lookup_by_text_fragment)
                    found = self._lookup(found, matched, 'unit_type', lookup_by_unit_type)
                    found = self._lookup(found, matched, 'unit_tier', lookup_by_unit_type)
                    found = self._lookup(found, matched, 'unit_name', lookup_by_unit_name)
                    if found.keys() == matched.keys():
                        data.loc[text, 'Mapped by pattern'] = replacement.format(**found)

        data = data.reset_index()[['Text', 'Count', 'Mapped by text', 'Mapped by pattern', 'File']].drop_duplicates(subset=['Text'])
        return data


class MapByPatternZhtw(MapByPatternZhcn):
    step_no = 23
    lang_col = 'zh-tw'


class MapByKeyZhcn(Step):
    step_no = 24
    lang_col = 'zh-cn'

    def __init__(self, trom_eng, vanilla_translations, trom_zh, lookup_by_key, map_by_pattern_zh):
        self.trom_eng = trom_eng
        self.vanilla_translations = vanilla_translations
        self.trom_zh = trom_zh
        self.lookup_by_key = lookup_by_key
        self.map_by_pattern_zh = map_by_pattern_zh

    def _run(self):
        data = self.trom_eng.data.fillna('').rename({'Text': 'English'}, axis=1).copy()
        vanilla = self.vanilla_translations.data.drop_duplicates(subset=['Key'])
        vanilla = vanilla[['Key', self.lang_col]].rename({self.lang_col: 'Vanilla'}, axis=1)
        trom_zh = self.trom_zh.data.drop_duplicates(subset=['Key'])[['Key', 'Text']]
        lookup_by_key = self.lookup_by_key.data[['Key', self.lang_col]].rename({self.lang_col: 'Override by key'}, axis=1)
        data = data.merge(lookup_by_key, on='Key', how='left')
        data = data.merge(vanilla, on='Key', how='left')
        data = data.merge(trom_zh.rename({'Text': '2.2 translation'}, axis=1), on='Key', how='left')
        data['Short Key'] = data['Key'].str.rsplit('_', n=1).str[0]
        data = data.merge(trom_zh.rename({'Text': '2.2 translation shortened key', 'Key': 'Short Key'}, axis=1), on='Short Key', how='left')

        data['Text'] = pd.NA
        data['Source'] = 'Missing'
        map_by_pattern_zh = self.map_by_pattern_zh.data[['Text', 'Mapped by text', 'Mapped by pattern']]
        data = data.merge(map_by_pattern_zh, left_on='English', right_on='Text', how='left')
        for col in [
            '2.2 translation shortened key',
            '2.2 translation',
            'Vanilla',
            'Mapped by text',
            'Mapped by pattern',
            'Override by key',
        ]:
            data.loc[~pd.isna(data[col]), ['Text']] = data[col]
            data.loc[~pd.isna(data[col]), ['Source']] = col
        data['File'] = data['File'].str.replace('text/_hvo/', '')
        data = data.reset_index()[[
            'Key', 'Text', 'Tooltip', 'English', 'Override by key',
            'Mapped by text', 'Mapped by pattern', 'Vanilla', '2.2 translation', '2.2 translation shortened key',
            'Source', 'Short Key', 'File',
        ]]
        return data


class MapByKeyZhtw(MapByKeyZhcn):
    step_no = 25
    lang_col = 'zh-tw'


class FinalZhcn(LocOutput):
    step_no = 26
    lang_col = 'zh-cn'

    def __init__(self, map_by_key_zh):
        self.map_by_key_zh = map_by_key_zh

    def _run(self):
        data = self.map_by_key_zh.data.copy()
        data.loc[data['Source'] == 'Missing', 'Text'] = data['English']
        data = data[['Key', 'Text', 'Tooltip']]
        return data


class FinalZhtw(FinalZhcn):
    step_no = 27
    lang_col = 'zh-tw'


class RegenLookupByText(Step):
    step_no = 28

    def __init__(self, map_by_key_zhcn, map_by_key_zhtw):
        self.map_by_key_zhcn = map_by_key_zhcn
        self.map_by_key_zhtw = map_by_key_zhtw

    def _run(self):
        new_lookup = pd.concat([
            zh.loc[
                zh.Source.isin(['Missing', 'Mapped by text']), ['English', from_col]
            ].rename({
                'English': 'eng',
                from_col: to_col,
            }, axis=1).drop_duplicates(subset='eng').set_index('eng')
            for zh, from_col, to_col in [
                (self.map_by_key_zhcn.data, 'Text', 'zh-cn'),
                (self.map_by_key_zhtw.data, 'Text', 'zh-tw'),
                (self.map_by_key_zhtw.data, 'File', 'File'),
            ]
        ], axis=1).sort_index().reset_index()
        new_lookup = new_lookup.loc[new_lookup['eng'] != '']
        return new_lookup


class MissingZhcn(Step):
    step_no = 40

    def __init__(self, map_by_key_zh):
        self.map_by_key_zh = map_by_key_zh

    def _run(self):
        data = self.map_by_key_zh.data
        data = data.loc[(data['Source'].isin([
            'Missing',
            # '2.2 translation shortened key',
            # '2.2 translation',
        ])) & (data['Text'] != '')]
        return data


class MissingZhtw(MissingZhcn):
    step_no = 41


class Comparison(Step):
    step_no = 42

    def __init__(self, trom_eng, vanilla_zhcn, vanilla_zhtw, map_by_key_zhcn, map_by_key_zhtw, trom_zhcn, trom_zhtw, procrastinator_zhcn):
        self.trom_eng = trom_eng
        self.vanilla_zhcn = vanilla_zhcn
        self.vanilla_zhtw = vanilla_zhtw
        self.map_by_key_zhcn = map_by_key_zhcn
        self.map_by_key_zhtw = map_by_key_zhtw
        self.trom_zhcn = trom_zhcn
        self.trom_zhtw = trom_zhtw
        self.procrastinator_zhcn = procrastinator_zhcn

    def _run(self):
        cmp = pd.concat([
            step.data.drop_duplicates(subset='Key').set_index('Key').rename({from_col: to_col}, axis=1)[to_col]
            for step, from_col, to_col in [
                (self.trom_eng, 'Text', 'English'),
                (self.map_by_key_zhcn, 'Text', 'MappedZhcn'),
                (self.map_by_key_zhcn, 'Source', 'SourceZhcn'),
                (self.vanilla_zhcn, 'Text', 'VanillaZhcn'),
                (self.trom_zhcn, 'Text', 'PikaManZhcn'),
                (self.procrastinator_zhcn, 'Text', 'ProcrastinatorZhcn'),
                (self.map_by_key_zhtw, 'Text', 'MappedZhtw'),
                (self.map_by_key_zhtw, 'Source', 'SourceZhtw'),
                (self.vanilla_zhtw, 'Text', 'VanillaZhtw'),
                (self.trom_zhtw, 'Text', 'PikaManZhtw'),
            ]
        ], axis=1).loc[self.trom_eng.data['Key'].drop_duplicates()].reset_index()
        cmp = cmp[(cmp['MappedZhcn'] != cmp['ProcrastinatorZhcn']) & (~cmp['ProcrastinatorZhcn'].isin([pd.NA, '', '尚未翻译']))]
        return cmp


def main():
    vanilla_zhcn = VanillaZhcn()
    vanilla_zhtw = VanillaZhtw()
    vanilla_eng = VanillaEng()
    trom_zhcn = TromZhcn()
    trom_zhtw = TromZhtw()
    trom_eng = TromEng()
    procrastinator_zhcn = ProcrastinatorZhcn()
    key_map_override = TromVanillaKeyOverride()
    vanilla_translations = VanillaTranslations(trom_eng, key_map_override, vanilla_eng, vanilla_zhtw, vanilla_zhcn)
    lookup_by_text = LookupByText()
    lookup_by_key = LookupByKey()
    lookup_by_pattern = LookupByPattern()
    lookup_by_unit_name = LookupByUnitName()
    lookup_by_unit_type = LookupByUnitType()
    lookup_by_text_fragment = LookupByTextFragment()
    map_by_pattern_zhcn = MapByPatternZhcn(trom_eng, vanilla_translations, lookup_by_text, lookup_by_pattern, lookup_by_unit_name, lookup_by_unit_type, lookup_by_text_fragment)
    map_by_pattern_zhtw = MapByPatternZhtw(trom_eng, vanilla_translations, lookup_by_text, lookup_by_pattern, lookup_by_unit_name, lookup_by_unit_type, lookup_by_text_fragment)
    map_by_key_zhcn = MapByKeyZhcn(trom_eng, vanilla_translations, trom_zhcn, lookup_by_key, map_by_pattern_zhcn)
    map_by_key_zhtw = MapByKeyZhtw(trom_eng, vanilla_translations, trom_zhtw, lookup_by_key, map_by_pattern_zhtw)
    final_zhcn = FinalZhcn(map_by_key_zhcn)
    final_zhtw = FinalZhtw(map_by_key_zhtw)
    regen_lookup_by_text = RegenLookupByText(map_by_key_zhcn, map_by_key_zhtw)
    missing_zhcn = MissingZhcn(map_by_key_zhcn)
    missing_zhtw = MissingZhtw(map_by_key_zhtw)
    comparison = Comparison(trom_eng, vanilla_zhcn, vanilla_zhtw, map_by_key_zhcn, map_by_key_zhtw, trom_zhcn, trom_zhtw, procrastinator_zhcn)

    for step in [
        vanilla_zhcn,
        vanilla_zhtw,
        vanilla_eng,
        trom_zhcn,
        trom_zhtw,
        trom_eng,
        procrastinator_zhcn,
        key_map_override,
        vanilla_translations,
        lookup_by_text,
        lookup_by_key,
        lookup_by_pattern,
        lookup_by_unit_name,
        lookup_by_unit_type,
        lookup_by_text_fragment,
        map_by_pattern_zhcn,
        map_by_pattern_zhtw,
        map_by_key_zhcn,
        map_by_key_zhtw,
        final_zhcn,
        final_zhtw,
        regen_lookup_by_text,
        missing_zhcn,
        missing_zhtw,
        comparison,
    ]:
        step.run()
        step.save()


if __name__ == '__main__':
    with pd.option_context(
        'display.width', 200,
        'display.max_columns', None,
    ):
        main()
