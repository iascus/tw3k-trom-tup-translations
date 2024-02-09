import glob
import logging
import os
import re
from abc import abstractmethod

import numpy as np
import pandas as pd
from opencc import OpenCC

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Step:
    step_no = None
    data = None
    dependencies = {}

    def __init__(self, results):
        for key, cls in self.dependencies.items():
            setattr(self, key, results[cls])

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


class InputCsvFile(Step):
    def _run(self):
        return self.load_file('in')


class LookupFile(Step):
    def _run(self):
        return self.load_file('lookup')


class InputTsvFiles(Step):
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


class VanillaZhcn(InputCsvFile):
    step_no = 1


class VanillaZhtw(InputCsvFile):
    step_no = 2


class VanillaEng(InputCsvFile):
    step_no = 3


class PikaManZhcn(InputTsvFiles):
    step_no = 4
    dir_path = 'PikaManZhcn'


class PikaManZhtw(InputTsvFiles):
    step_no = 5
    dir_path = 'PikaManZhtw'


class TromEng(InputTsvFiles):
    step_no = 6
    dir_path = 'Trom3.9e'


class ProcrastinatorZhcn(InputTsvFiles):
    step_no = 7
    dir_path = 'ProcrastinatorZhcn'

    def _run(self):
        data = super()._run()
        data.loc[data['Text'].isin(['', '尚未翻译']), 'Text'] = np.nan
        return data


class ProcrastinatorZhtw(Step):
    step_no = 8

    dependencies = {
        'procrastinator_zhcn': ProcrastinatorZhcn,
    }

    def _run(self):
        data = self.procrastinator_zhcn.data.copy()
        cc = OpenCC('s2t')
        data['Text'] = data['Text'].apply(lambda string: cc.convert(string) if not pd.isna(string) else string)
        return data


class IascusZhcn(InputCsvFile):
    step_no = 9


class IascusZhtw(InputCsvFile):
    step_no = 10


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


class LookupBySkill(LookupFile):
    step_no = 17


class LookupByTextFragment(LookupFile):
    step_no = 18


class VanillaTranslations(Step):
    step_no = 21

    dependencies = {
        'trom_eng': TromEng,
        'key_map_override': TromVanillaKeyOverride,
        'vanilla_eng': VanillaEng,
        'vanilla_zhtw': VanillaZhtw,
        'vanilla_zhcn': VanillaZhcn,
    }

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

    dependencies = {
        'trom_eng': TromEng,
        'vanilla_translations': VanillaTranslations,
        'lookup_by_text': LookupByText,
        'lookup_by_pattern': LookupByPattern,
        'lookup_by_unit_name': LookupByUnitName,
        'lookup_by_unit_type': LookupByUnitType,
        'lookup_by_skill': LookupBySkill,
        'lookup_by_text_fragment': LookupByTextFragment,
    }

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
        lookup_by_text = lookup_by_text.set_index('eng')[[self.lang_col]].rename({self.lang_col: 'MappedByText'}, axis=1)
        data = data.merge(lookup_by_text, left_index=True, right_index=True, how='left')
        data['MappedByPattern'] = pd.Series(pd.NA, dtype='string')

        lookup_by_text_fragment = self.lookup_by_text_fragment.data.set_index('eng')[self.lang_col].to_dict()
        lookup_by_pattern = self.lookup_by_pattern.data.set_index(['KeyPattern', 'TextPattern'])[[self.lang_col]]
        lookup_by_unit_type = self.lookup_by_unit_type.data.set_index('Text')[self.lang_col].to_dict()
        lookup_by_unit_name = self.lookup_by_unit_name.data.set_index('Text')[self.lang_col].to_dict()
        lookup_by_skill = self.lookup_by_skill.data.set_index('Text')[self.lang_col].to_dict()
        lookup_by_text_vanilla = self.vanilla_translations.data[['Text', self.lang_col]].dropna().set_index('Text')[self.lang_col].to_dict()
        lookup_by_text_vanilla.update(lookup_by_text.to_dict()['MappedByText'])
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
                    found = self._lookup(found, matched, 'skill', lookup_by_skill)
                    if found.keys() == matched.keys():
                        data.loc[text, 'MappedByPattern'] = replacement.format(**found)

        data = data.reset_index()[['Text', 'Count', 'MappedByText', 'MappedByPattern', 'File']].drop_duplicates(subset=['Text'])
        return data


class MapByPatternZhtw(MapByPatternZhcn):
    step_no = 23
    lang_col = 'zh-tw'


class MapByKeyZhcn(Step):
    step_no = 24
    lang_col = 'zh-cn'

    dependencies = {
        'trom_eng': TromEng,
        'vanilla_translations': VanillaTranslations,
        'pikaman_zh': PikaManZhcn,
        'procrastinator_zh': ProcrastinatorZhcn,
        'lookup_by_key': LookupByKey,
        'map_by_pattern_zh': MapByPatternZhcn,
    }

    def _run(self):
        data = self.trom_eng.data.fillna('').rename({'Text': 'English'}, axis=1).copy()
        vanilla = self.vanilla_translations.data.drop_duplicates(subset=['Key'])
        vanilla = vanilla[['Key', self.lang_col]].rename({self.lang_col: 'Vanilla'}, axis=1)
        pikaman_zh = self.pikaman_zh.data.drop_duplicates(subset=['Key'])[['Key', 'Text']]
        procrastinator_zh = self.procrastinator_zh.data.drop_duplicates(subset=['Key'])[['Key', 'Text']]
        lookup_by_key = self.lookup_by_key.data[['Key', self.lang_col]].rename({self.lang_col: 'OverrideByKey'}, axis=1)
        map_by_pattern_zh = self.map_by_pattern_zh.data[['Text', 'MappedByText', 'MappedByPattern']].rename({'Text': 'English'}, axis=1)

        data = data.merge(lookup_by_key, on='Key', how='left')
        data = data.merge(vanilla, on='Key', how='left')
        data = data.merge(pikaman_zh.rename({'Text': 'PikaMan'}, axis=1), on='Key', how='left')
        data['Short Key'] = data['Key'].str.rsplit('_', n=1).str[0]
        data = data.merge(pikaman_zh.rename({'Text': 'PikaManShortenedKey', 'Key': 'Short Key'}, axis=1), on='Short Key', how='left')
        data = data.merge(procrastinator_zh.rename({'Text': 'Procrastinator'}, axis=1), on='Key', how='left')
        data['Text'] = pd.NA
        data['Source'] = 'Missing'
        data = data.merge(map_by_pattern_zh, on='English', how='left')
        for col in [
            'PikaManShortenedKey',
            'PikaMan',
            'Procrastinator',
            'Vanilla',
            'MappedByText',
            'MappedByPattern',
            'OverrideByKey',
        ]:
            to_update = (~pd.isna(data[col]) & (data[col] != ''))
            data.loc[to_update, ['Source']] = col
            data.loc[to_update, ['Text']] = data[col]

        data['File'] = data['File'].str.replace('text/_hvo/', '')
        data = data.reset_index()[[
            'Key', 'Text', 'Tooltip', 'English', 'OverrideByKey',
            'MappedByText', 'MappedByPattern', 'Vanilla',
            'PikaMan', 'PikaManShortenedKey',
            'Source', 'Short Key', 'File',
        ]]
        return data


class MapByKeyZhtw(MapByKeyZhcn):
    step_no = 25
    lang_col = 'zh-tw'

    dependencies = {
        'trom_eng': TromEng,
        'vanilla_translations': VanillaTranslations,
        'pikaman_zh': PikaManZhtw,
        'procrastinator_zh': ProcrastinatorZhtw,
        'lookup_by_key': LookupByKey,
        'map_by_pattern_zh': MapByPatternZhtw,
    }


class FinalZhcn(LocOutput):
    step_no = 26
    lang_col = 'zh-cn'

    dependencies = {
        'map_by_key_zh': MapByKeyZhcn,
    }

    def _run(self):
        data = self.map_by_key_zh.data.copy()
        data.loc[data['Source'] == 'Missing', 'Text'] = data['English']
        data = data[['Key', 'Text', 'Tooltip']]
        return data


class FinalZhtw(FinalZhcn):
    step_no = 27
    lang_col = 'zh-tw'

    dependencies = {
        'map_by_key_zh': MapByKeyZhtw,
    }


class RegenLookupByText(Step):
    step_no = 28

    dependencies = {
        'map_by_key_zhcn': MapByKeyZhcn,
        'map_by_key_zhtw': MapByKeyZhtw,
    }

    def _run(self):
        new_lookup = pd.concat([
            zh.loc[
                zh.Source.isin(['Missing', 'MappedByText']), ['English', from_col]
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

    dependencies = {
        'map_by_key_zh': MapByKeyZhcn,
    }

    def _run(self):
        data = self.map_by_key_zh.data
        data = data.loc[(data['Source'].isin([
            'Missing',
            # 'PikaManShortenedKey',
            # 'PikaMan',
        ])) & (data['Text'] != '')]
        return data


class MissingZhtw(MissingZhcn):
    step_no = 41

    dependencies = {
        'map_by_key_zh': MapByKeyZhtw,
    }


class Comparison(Step):
    step_no = 42

    dependencies = {
        'trom_eng': TromEng,
        'vanilla_zhcn': VanillaZhcn,
        'vanilla_zhtw': VanillaZhtw,
        'map_by_key_zhcn': MapByKeyZhcn,
        'map_by_key_zhtw': MapByKeyZhtw,
        'pikaman_zhcn': PikaManZhcn,
        'pikaman_zhtw': PikaManZhtw,
        'procrastinator_zhcn': ProcrastinatorZhcn,
        'procrastinator_zhtw': ProcrastinatorZhtw,
        'iascus_zhcn': IascusZhcn,
        'iascus_zhtw': IascusZhtw,
    }

    def _run(self):
        cmp = pd.concat([
            step.data.drop_duplicates(subset='Key').set_index('Key').rename({from_col: to_col}, axis=1)[to_col]
            for step, from_col, to_col in [
                (self.trom_eng, 'Text', 'English'),
                (self.map_by_key_zhcn, 'Text', 'MappedZhcn'),
                (self.map_by_key_zhcn, 'Source', 'SourceZhcn'),
                (self.map_by_key_zhcn, 'Vanilla', 'VanillaZhcn'),
                (self.iascus_zhcn, 'Text', 'IascusZhcn'),
                (self.procrastinator_zhcn, 'Text', 'ProcrastinatorZhcn'),
                (self.map_by_key_zhtw, 'Text', 'MappedZhtw'),
                (self.map_by_key_zhtw, 'Source', 'SourceZhtw'),
                (self.map_by_key_zhtw, 'Vanilla', 'VanillaZhtw'),
                (self.iascus_zhtw, 'Text', 'IascusZhtw'),
                (self.procrastinator_zhtw, 'Text', 'ProcrastinatorZhtw'),
            ]
        ], axis=1).loc[self.trom_eng.data['Key'].drop_duplicates()].reset_index()

        cmp = cmp[
            cmp['Key'].str.startswith('character_skills_localised_')
        ]
        cmp.insert(7, 'DiffZhcn', cmp['ProcrastinatorZhcn'] != cmp['MappedZhcn'])
        cmp.insert(8, 'DiffZhtw', cmp['ProcrastinatorZhtw'] != cmp['MappedZhtw'])
        cmp['SkillKey'] = cmp['Key'].str.replace(
            re.compile('character_skills_localised_([a-z]+)_(.*)'), r'\2_\1', regex=True
        )
        cmp = cmp.drop('Key', axis=1).sort_values('SkillKey', ascending=False).drop_duplicates()
        cmp = cmp.loc[~cmp['English'].isin([
            'First Characteristic',
            'Someone who stands out among the people.',
            'Second Characteristic',
            'A remarkable talent.',
            'Third Characteristic',
            'Every person has their own path.',
        ])]
        # cmp = cmp[~cmp['SourceZhtw'].isin(['Vanilla', 'MappedByPattern'])]
        return cmp


def main():
    steps = {
        cls: None
        for cls in [
            VanillaZhcn,
            VanillaZhtw,
            VanillaEng,
            PikaManZhcn,
            PikaManZhtw,
            TromEng,
            ProcrastinatorZhcn,
            ProcrastinatorZhtw,
            IascusZhcn,
            IascusZhtw,
            TromVanillaKeyOverride,
            VanillaTranslations,
            LookupByText,
            LookupByKey,
            LookupByPattern,
            LookupByUnitName,
            LookupByUnitType,
            LookupBySkill,
            LookupByTextFragment,
            MapByPatternZhcn,
            MapByPatternZhtw,
            MapByKeyZhcn,
            MapByKeyZhtw,
            FinalZhcn,
            FinalZhtw,
            RegenLookupByText,
            MissingZhcn,
            MissingZhtw,
            Comparison,
        ]
    }
    for cls in list(steps.keys()):
        step = cls(steps)
        steps[cls] = step
        step.run()
        step.save()


if __name__ == '__main__':
    with pd.option_context(
        'display.width', 200,
        'display.max_columns', None,
    ):
        main()
