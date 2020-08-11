import pandas as pd
import re
from typing import List, Dict


class DataPreprocessor:

    def __init__(self,
                 data_path: str,
                 context_window_size: int,
                 ):
        self.data = pd.read_csv(data_path, sep='\t')
        self.context_window_size = context_window_size
        self.allowed_character = list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя' +
                                      'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'.upper() +
                                      '0123456789' +
                                      ' .,?!:()'
                                      )

    def get_data_list(self) -> List:
        data = self._basic_preprocess(self.data)
        data = data.apply(self._concat_consecutive_message)
        data = data.apply(lambda x: self.preprocess_strings(x, self.allowed_character))
        data = data.apply(self._get_windows)
        return sum(list(data), [])

    def _basic_preprocess(self, data: pd.DataFrame) -> pd.Series:
        data = data['dialogue'].apply(lambda x: re.sub('<br />', '', x))
        data = data.apply(lambda x: list(map(self._rename_users, x.split('</span>'))))
        data = data.apply(lambda x: x[:-1])
        return data

    @staticmethod
    def preprocess_strings(target: List[str], allowed_characters: List[str]):
        result = []
        allow_chars_string = ''.join(allowed_characters)
        for string in target:
            step = re.sub(fr'[^{allow_chars_string}]', ' ', string)
            step = re.sub(' +', ' ', step)
            result.append(step)
        return result

    @staticmethod
    def _rename_users(target_string: str):
        target_string = re.sub('<span class=participant_1>Пользователь 1:', '<user1>', target_string)
        target_string = re.sub('<span class=participant_2>Пользователь 2:', '<user2>', target_string)
        return target_string

    @staticmethod
    def _concat_consecutive_message(messages: List[str]) -> List[str]:
        start = 0
        first_message = messages[0]
        while messages[start][:7] == messages[start + 1][:7]:
            first_message += messages[start + 1][7:]
            start += 1
        result = [first_message]
        prev_user = messages[0][:7]
        conseq_message = ''
        conseq = False
        for i in range(start + 1, len(messages)):
            cur_user = messages[i][:7]

            if cur_user != prev_user:
                if conseq:
                    result[-1] = conseq_message
                    result.append(messages[i])
                    conseq_message = messages[i]
                    conseq = False
                else:
                    conseq_message = messages[i]
                    result.append(conseq_message)
            else:
                conseq_message += messages[i][7:]
                conseq = True
            prev_user = cur_user
        if i == (len(messages) - 1) and conseq:
            result.append(conseq_message)
        result = list(map(lambda x: re.sub(r'<user\d> ', '', x), result))
        return result

    def _get_windows(self, messages: List[str]) -> List[Dict]:
        windows = []
        k = self.context_window_size + 1
        for i in range(k - 1, 0, -1):
            window = [''] * i
            for j in range(k - i):
                window.append(messages[j])
            windows.append(
                {
                    'context': window[:-1],
                    'target': window[-1]
                }
            )
        for i in range(len(messages) - k + 1):
            window = messages[i: i + k]

            windows.append(
                {
                    'context': window[:-1],
                    'target': window[-1]
                }
            )
        return windows
