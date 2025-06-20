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
        self.data.to_csv(self.out_filepath, index=False, lineterminator='\n')
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

    def blank_wip(self, data):
        data.loc[data['Text'].isin([
            '',
            '尚未翻译',
            '未完成',
            r'未完成\\n\\n（请于MTU的Discord里TROM频道提出建议）',
            r'未完成\\n\\n（請於MTU的Discord裏TROM頻道提出建議）',
        ]), 'Text'] = np.nan
        return data


class InputCsvFile(Step):
    def _run(self):
        data = self.load_file('in')
        data = self.blank_wip(data)
        return data


class LookupFile(Step):
    def _run(self):
        return self.load_file('lookup')


class InputTsvFiles(Step):
    dir_path = None

    def _run(self):
        dfs = []
        for filepath in sorted(glob.glob(os.path.join('csv', 'in', self.dir_path, '**', '*.tsv'), recursive=True)):
            with open(filepath, encoding='utf-8') as tsv_file:
                df = pd.read_csv(tsv_file, sep='\t', skiprows=[1])
                df.columns = ['Key', 'Text', 'Tooltip']
                df['File'] = os.path.basename(filepath)
                dfs.append(df)
        data = pd.concat(dfs).dropna(subset='Key').drop_duplicates(subset='Key').sort_values('Key')
        data = self.blank_wip(data)
        data = data[~data['Key'].str.endswith('----')]
        return data


class LocOutput(Step):

    tsv_filename = '!@hv_TEXT.tsv'
    lang_col = None

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


class VanillaZhcn(InputTsvFiles):
    step_no = 1
    dir_path = 'VanillaZhcn'


class VanillaZhtw(InputTsvFiles):
    step_no = 2
    dir_path = 'VanillaZhtw'


class VanillaEng(InputTsvFiles):
    step_no = 3
    dir_path = 'VanillaEng'


class TromEng(InputTsvFiles):
    step_no = 4
    dir_path = 'TromEng'


class MtuZhcn(InputTsvFiles):
    step_no = 11
    dir_path = 'MtuZhcn'


class MtuZhtw(InputTsvFiles):
    step_no = 12
    dir_path = 'MtuZhtw'


class LeiwaiZhcn(InputTsvFiles):
    step_no = 13
    dir_path = 'LeiwaiZhcn'


class LeiwaiZhtw(Step):
    step_no = 14
    dir_path = 'LeiwaiZhcn'

    dependencies = {
        'leiwai_zhcn': LeiwaiZhcn,
    }

    def _run(self):
        data = self.leiwai_zhcn.data.copy()
        cc = OpenCC('s2t')
        data['Text'] = data['Text'].apply(lambda string: cc.convert(string) if not pd.isna(string) else string)
        return data


class PikaManZhcn(InputTsvFiles):
    step_no = 21
    dir_path = 'PikaManZhcn'


class PikaManZhtw(InputTsvFiles):
    step_no = 22
    dir_path = 'PikaManZhtw'


class ProcrastinatorZhcn(InputTsvFiles):
    step_no = 23
    dir_path = 'ProcrastinatorZhcn'


class ProcrastinatorZhtw(Step):
    step_no = 24

    dependencies = {
        'procrastinator_zhcn': ProcrastinatorZhcn,
    }

    def _run(self):
        data = self.procrastinator_zhcn.data.copy()
        cc = OpenCC('s2t')
        data['Text'] = data['Text'].apply(lambda string: cc.convert(string) if not pd.isna(string) else string)
        return data


class IascusZhcn(InputCsvFile):
    step_no = 25


class IascusZhtw(InputCsvFile):
    step_no = 26


class LookupByVanilla(Step):
    step_no = 31

    dependencies = {
        'vanilla_eng': VanillaEng,
        'vanilla_zhtw': VanillaZhtw,
        'vanilla_zhcn': VanillaZhcn,
    }

    def _run(self):
        vanilla_translations = pd.concat([
            df.data.set_index('Key')[['Text']].rename({'Text': text_col}, axis=1)
            for df, text_col in [
                (self.vanilla_eng, 'Text'), (self.vanilla_zhtw, 'zh-tw'), (self.vanilla_zhcn, 'zh-cn')
            ]] + [self.vanilla_eng.data.set_index('Key')[['File']]], axis=1
        ).dropna(subset=['Text']).drop_duplicates(['Text', 'zh-cn', 'zh-tw']).sort_values(['File', 'Text', 'zh-cn', 'zh-tw'])
        vanilla_translations['Duplicated'] = vanilla_translations.duplicated(subset=['File', 'Text'], keep=False)
        vanilla_translations = vanilla_translations.reset_index().rename({'Key': 'VanillaKey'}, axis=1)[[
            'File', 'Text', 'VanillaKey', 'zh-cn', 'zh-tw', 'Duplicated',
        ]]
        vanilla_translations = vanilla_translations.sort_values(['Duplicated', 'File', 'Text'])
        return vanilla_translations


class LookupByText(LookupFile):
    step_no = 32


class LookupByKey(LookupFile):
    step_no = 33


class LookupByPattern(LookupFile):
    step_no = 34


class LookupFilePlusVanilla(LookupFile):
    dependencies = {
        'lookup_by_vanilla': LookupByVanilla,
    }


    def _run(self):
        lookup = self.load_file('lookup')
        vanilla = self.lookup_by_vanilla.data.copy()
        vanilla = vanilla.loc[
            vanilla.File.isin([f'{file}__.loc.tsv' for file in self.vanilla_files]),
            ['Text', 'zh-cn', 'zh-tw'],
        ].rename({'Text': 'eng'}, axis=1).drop_duplicates()
        lookup['Source'] = 'Lookup'
        vanilla['Source'] = 'Vanilla'
        lookup = pd.concat([vanilla, lookup]).sort_values(lookup.columns.tolist())
        lookup['Duplicated'] = lookup.duplicated(subset=['eng'], keep=False)
        lookup = lookup.sort_values(['Duplicated', 'eng'])
        return lookup


class LookupByUnitName(LookupFilePlusVanilla):
    step_no = 35
    vanilla_files = [
        'land_units',
        'ui_unit_groupings',
        'ui_unit_group_parents',
    ]


class LookupByUnitType(LookupFilePlusVanilla):
    step_no = 36
    vanilla_files = [
        'ui_unit_group_parents',
        'ui_unit_groupings',
    ]


class LookupBySkill(LookupFile):
    step_no = 37


class LookupByCharacter(LookupFilePlusVanilla):
    step_no = 38
    vanilla_files = [
        'factions',
        'land_units',
    ]


class LookupByRegion(LookupFilePlusVanilla):
    step_no = 39
    vanilla_files = [
        'campaign_battle_presets',
        'regions',
    ]


class LookupByTextFragment(LookupFile):
    step_no = 40


class VanillaTranslations(Step):
    step_no = 41

    dependencies = {
        'trom_eng': TromEng,
        'lookup_by_vanilla': LookupByVanilla,
    }

    def _run(self):
        trom_eng = self.trom_eng.data.dropna(subset='Text').set_index('Text')
        vanilla_translations = self.lookup_by_vanilla.data.reset_index()
        vanilla_translations = vanilla_translations.drop_duplicates(subset='Text', keep=False).set_index('Text')
        data = trom_eng.join(
            vanilla_translations[['VanillaKey', 'zh-cn', 'zh-tw']], how='left'
        ).reset_index()[['Key', 'Text', 'Tooltip', 'VanillaKey', 'zh-cn', 'zh-tw', 'File']]
        return data


class MapByPatternZhcn(Step):
    step_no = 42
    lang_col = 'zh-cn'

    dependencies = {
        'trom_eng': TromEng,
        'vanilla_translations': VanillaTranslations,
        'lookup_by_text': LookupByText,
        'lookup_by_pattern': LookupByPattern,
        'lookup_by_unit_name': LookupByUnitName,
        'lookup_by_unit_type': LookupByUnitType,
        'lookup_by_skill': LookupBySkill,
        'lookup_by_character': LookupByCharacter,
        'lookup_by_region': LookupByRegion,
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

    def _get_lookup_dict(self, step):
        df = step.data
        if 'Source' in df.columns:
            df = df.sort_values(['Source', 'eng']).drop_duplicates(subset='eng')
        return df.set_index('eng')[self.lang_col].to_dict()

    def _run(self):
        data = self.trom_eng.data.fillna('').copy()
        data = data.groupby(['Text']).agg(Tooltip=('Tooltip', 'first'), File=('File', 'first'), Count=('Key', 'count'), Key=('Key', 'first'))
        lookup_by_text = self.lookup_by_text.data[~pd.isna(self.lookup_by_text.data[self.lang_col])]
        lookup_by_text = lookup_by_text.set_index('eng')[[self.lang_col]].rename({self.lang_col: 'MappedByText'}, axis=1)
        data = data.merge(lookup_by_text, left_index=True, right_index=True, how='left')
        data['MappedByPattern'] = pd.Series(pd.NA, dtype='string')

        lookup_by_pattern = self.lookup_by_pattern.data.set_index(['KeyPattern', 'TextPattern'])[[self.lang_col]]
        lookup_by_text_fragment = self._get_lookup_dict(self.lookup_by_text_fragment)
        lookup_by_unit_type = self._get_lookup_dict(self.lookup_by_unit_type)
        lookup_by_unit_name = self._get_lookup_dict(self.lookup_by_unit_name)
        lookup_by_skill = self._get_lookup_dict(self.lookup_by_skill)
        lookup_by_character = self._get_lookup_dict(self.lookup_by_character)
        lookup_by_region = self._get_lookup_dict(self.lookup_by_region)
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
                    found = self._lookup(found, matched, 'character', lookup_by_character)
                    found = self._lookup(found, matched, 'region', lookup_by_region)
                    if found.keys() == matched.keys():
                        data.loc[text, 'MappedByPattern'] = replacement.format(**found)

        data = data.reset_index()[['Text', 'Count', 'MappedByText', 'MappedByPattern', 'File']].drop_duplicates(subset=['Text'])
        return data


class MapByPatternZhtw(MapByPatternZhcn):
    step_no = 43
    lang_col = 'zh-tw'


class MapByKeyZhcn(Step):
    step_no = 44
    lang_col = 'zh-cn'

    dependencies = {
        'trom_eng': TromEng,
        'vanilla_translations': VanillaTranslations,
        'mtu_zh': MtuZhcn,
        'leiwai_zh': LeiwaiZhcn,
        'pikaman_zh': PikaManZhcn,
        'procrastinator_zh': ProcrastinatorZhcn,
        'iascus_zh': IascusZhcn,
        'lookup_by_key': LookupByKey,
        'map_by_pattern_zh': MapByPatternZhcn,
    }

    def _run(self):
        data = self.trom_eng.data.fillna('').rename({'Text': 'English'}, axis=1).copy()
        vanilla = self.vanilla_translations.data.drop_duplicates(subset=['Key'])
        vanilla = vanilla[['Key', self.lang_col]].rename({self.lang_col: 'Vanilla'}, axis=1)
        mtu_zh = self.mtu_zh.data.drop_duplicates(subset=['Key'])[['Key', 'Text']]
        leiwai_zh = self.leiwai_zh.data.drop_duplicates(subset=['Key'])[['Key', 'Text']]
        pikaman_zh = self.pikaman_zh.data.drop_duplicates(subset=['Key'])[['Key', 'Text']]
        procrastinator_zh = self.procrastinator_zh.data.drop_duplicates(subset=['Key'])[['Key', 'Text']]
        iascus_zh = self.iascus_zh.data.drop_duplicates(subset=['Key'])[['Key', 'Text']]
        lookup_by_key = self.lookup_by_key.data[['Key', self.lang_col]].rename({self.lang_col: 'OverrideByKey'}, axis=1)
        map_by_pattern_zh = self.map_by_pattern_zh.data[['Text', 'MappedByText', 'MappedByPattern']].rename({'Text': 'English'}, axis=1)

        data = data.merge(lookup_by_key, on='Key', how='left')
        data = data.merge(vanilla, on='Key', how='left')
        data = data.merge(mtu_zh.rename({'Text': 'Mtu'}, axis=1), on='Key', how='left')
        data = data.merge(leiwai_zh.rename({'Text': 'Leiwai'}, axis=1), on='Key', how='left')
        data = data.merge(pikaman_zh.rename({'Text': 'PikaMan'}, axis=1), on='Key', how='left')
        data = data.merge(procrastinator_zh.rename({'Text': 'Procrastinator'}, axis=1), on='Key', how='left')
        data = data.merge(iascus_zh.rename({'Text': 'Iascus'}, axis=1), on='Key', how='left')
        data = data.merge(map_by_pattern_zh, on='English', how='left')
        data = data.fillna('')
        data['Text'] = ''
        data['Source'] = 'Missing'
        for col in [
            'Mtu',
            'PikaMan',
            'Iascus',
            'Procrastinator',
            'Leiwai',
            'Vanilla',
            'MappedByText',
            'MappedByPattern',
            'OverrideByKey',
        ]:
            to_update = (
                (data[col] != '')
                & (
                    (data[col] != data['Text'])
                    | (
                        (data['Source'] == 'Iascus')
                        &
                        (col in ['Vanilla', 'MappedByText', 'MappedByPattern', 'OverrideByKey',])
                    )
                )
            )
            data.loc[to_update, ['Source']] = col
            data.loc[to_update, ['Text']] = data[col]
        data.loc[data['Source'] == 'Iascus', ['Source', 'Text']] = ['Missing', '']
        data['File'] = data['File'].str.replace('text/_hvo/', '')
        data = data.reset_index()[[
            'Key', 'Text', 'Tooltip', 'English', 'Source',
            'OverrideByKey', 'MappedByPattern', 'MappedByText', 'Iascus',
            'Vanilla', 'Leiwai', 'Procrastinator', 'PikaMan', 'Mtu',
            'File',
        ]]
        return data


class MapByKeyZhtw(MapByKeyZhcn):
    step_no = 45
    lang_col = 'zh-tw'

    dependencies = {
        'trom_eng': TromEng,
        'vanilla_translations': VanillaTranslations,
        'mtu_zh': MtuZhtw,
        'leiwai_zh': LeiwaiZhtw,
        'pikaman_zh': PikaManZhtw,
        'procrastinator_zh': ProcrastinatorZhtw,
        'iascus_zh': IascusZhtw,
        'lookup_by_key': LookupByKey,
        'map_by_pattern_zh': MapByPatternZhtw,
    }


class FinalZhcn(LocOutput):
    step_no = 46
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
    step_no = 47
    lang_col = 'zh-tw'

    dependencies = {
        'map_by_key_zh': MapByKeyZhtw,
    }


class RegenLookupByText(Step):
    step_no = 51

    dependencies = {
        'map_by_key_zhcn': MapByKeyZhcn,
        'map_by_key_zhtw': MapByKeyZhtw,
    }

    def _run(self):
        new_lookup = pd.concat([
            zh.loc[
                (
                    self.map_by_key_zhcn.data.Source.isin(['Missing', 'MappedByText'])
                    | self.map_by_key_zhtw.data.Source.isin(['Missing', 'MappedByText'])
                ), ['English', from_col]
            ].rename({
                'English': 'eng',
                from_col: to_col,
            }, axis=1).drop_duplicates(subset='eng').set_index('eng')
            for zh, from_col, to_col in [
                (self.map_by_key_zhcn.data, 'Text', 'zh-cn'),
                (self.map_by_key_zhcn.data, 'Iascus', 'zh-cn-old'),
                (self.map_by_key_zhtw.data, 'Text', 'zh-tw'),
                (self.map_by_key_zhtw.data, 'Iascus', 'zh-tw-old'),
                (self.map_by_key_zhtw.data, 'File', 'File'),
            ]
        ], axis=1).reset_index().set_index(['File', 'eng']).sort_index().reset_index()
        new_lookup['zh-cn'] = np.where(new_lookup['zh-cn'] == '', new_lookup['zh-cn-old'], new_lookup['zh-cn'])
        new_lookup['zh-tw'] = np.where(new_lookup['zh-tw'] == '', new_lookup['zh-tw-old'], new_lookup['zh-tw'])
        new_lookup = new_lookup.loc[new_lookup['eng'] != ''].drop(['zh-cn-old', 'zh-tw-old'], axis=1)
        return new_lookup


class RegenLookupBySkill(Step):
    step_no = 52

    dependencies = {
        'map_by_key_zhcn': MapByKeyZhcn,
        'map_by_key_zhtw': MapByKeyZhtw,
    }

    def _run(self):
        keys = self.map_by_key_zhcn.data[['Key']]

        dfs = []
        for prefix in ['character_skills_localised_description_', 'character_skills_localised_name_',
                       'unit_abilities_onscreen_name_', 'unit_abilities_tooltip_text_',
                       'unit_ability_types_localised_description_', 'special_ability_phases_onscreen_name_']:
            skills = keys.loc[keys['Key'].str.startswith(prefix), ['Key']]
            skills['Skill'] = skills['Key'].str.replace(prefix, '')
            dfs.append(skills)
        skills = pd.concat(dfs)
        new_lookup = pd.concat([
            zh.set_index('Key').loc[skills['Key'], [from_col]].rename({from_col: to_col,}, axis=1)
            for zh, from_col, to_col in [
                (self.map_by_key_zhcn.data, 'Text', 'zh-cn'),
                (self.map_by_key_zhtw.data, 'Text', 'zh-tw'),
                (self.map_by_key_zhtw.data, 'English', 'eng'),
                (self.map_by_key_zhtw.data, 'File', 'File'),
                (skills, 'Skill', 'Skill'),
            ]
        ], axis=1)
        new_lookup = new_lookup.loc[new_lookup['eng'] != ''].sort_values(['File', 'Skill']).drop_duplicates(subset=['eng'])
        new_lookup = new_lookup.reset_index()[['Skill', 'eng', 'zh-cn', 'zh-tw', 'Key', 'File']]
        return new_lookup


class MissingZhcn(Step):
    step_no = 53

    dependencies = {
        'map_by_key_zh': MapByKeyZhcn,
    }

    def _run(self):
        data = self.map_by_key_zh.data
        data = data.loc[
            (data['Source'].isin([
                'Missing',
            ]))
            & (data['English'].fillna('') != '')
            # (~data['File'].str.contains('MTU'))
        ]
        return data


class MissingZhtw(MissingZhcn):
    step_no = 54

    dependencies = {
        'map_by_key_zh': MapByKeyZhtw,
    }


class Comparison(Step):
    step_no = 55

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
        'leiwai_zhcn': LeiwaiZhcn,
        'leiwai_zhtw': LeiwaiZhtw,
        'iascus_zhcn': IascusZhcn,
        'iascus_zhtw': IascusZhtw,
        'lookup_by_skill': LookupBySkill,
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

        cmp.insert(7, 'DiffZhcn', cmp['ProcrastinatorZhcn'] != cmp['MappedZhcn'])
        cmp.insert(8, 'DiffZhtw', cmp['ProcrastinatorZhtw'] != cmp['MappedZhtw'])
        # cmp = cmp.loc[cmp['ProcrastinatorZhcn'] != cmp['MappedZhcn']].dropna(subset='ProcrastinatorZhcn')
        skills = self.lookup_by_skill.data[['eng', 'Skill']].drop_duplicates('eng').set_index('eng')
        cmp = cmp.merge(skills, left_on='English', right_index=True, how='right')
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
            MtuZhcn,
            MtuZhtw,
            LeiwaiZhcn,
            LeiwaiZhtw,
            PikaManZhcn,
            PikaManZhtw,
            TromEng,
            ProcrastinatorZhcn,
            ProcrastinatorZhtw,
            IascusZhcn,
            IascusZhtw,
            LookupByVanilla,
            LookupByText,
            LookupByKey,
            LookupByPattern,
            LookupByUnitName,
            LookupByUnitType,
            LookupBySkill,
            LookupByCharacter,
            LookupByRegion,
            LookupByTextFragment,
            VanillaTranslations,
            MapByPatternZhcn,
            MapByPatternZhtw,
            MapByKeyZhcn,
            MapByKeyZhtw,
            FinalZhcn,
            FinalZhtw,
            RegenLookupByText,
            RegenLookupBySkill,
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
