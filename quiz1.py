# ========================================================================
# Copyright 2021 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import re
def normalize(text):

    # Function Tokenizer
    RE_TOK = re.compile(r'([",.]|n\'t|\'|\?|\!|\s+)')
    def tokenize_regex(text):
        prev_idx = 0
        tokens = []
        for m in RE_TOK.finditer(text):
            t = text[prev_idx:m.start()].strip()
            if t: tokens.append(t)
            t = m.group().strip()
            if t:
                if tokens and tokens[-1] in {'Mr', 'Ms'} and t == '.':
                    tokens[-1] = tokens[-1] + t
                else:
                    tokens.append(t)
            prev_idx = m.end()

        t = text[prev_idx:]
        if t:  tokens.append(t)
        return tokens

    # Function ParseInt
    def parse_int(string):
        NUM = {'zero': 0,
               'one': 1, 'a': 1, 'an': 1,
               'two': 2,
               'three': 3,
               'four': 4,
               'five': 5,
               'six': 6,
               'seven': 7,
               'eight': 8,
               'nine': 9,
               'ten': 10,
               'eleven': 11,
               'twelve': 12,
               'thirteen': 13,
               'fourteen': 14,
               'fifteen': 15,
               'sixteen': 16,
               'seventeen': 17,
               'eighteen': 18,
               'nineteen': 19,
               'twenty': 20,
               'thirty': 30,
               'forty': 40,
               'fifty': 50,
               'sixty': 60,
               'seventy': 70,
               'eighty': 80,
               'ninety': 90,
               }
        numbers = []
        for token in string.replace('-', ' ').split(' '):
            if token.lower() in NUM:
                numbers.append(NUM[token.lower()])
            elif token.lower() == 'hundred':
                #numbers[-1] *= 100
                numbers = [x * 100 for x in numbers]
            elif token.lower() == 'thousand':
                numbers = [x * 1000 for x in numbers]
            elif token.lower() == 'million':
                numbers = [x * 1000000 for x in numbers]
            elif token.lower() == 'billion':
                numbers = [x * 1000000000 for x in numbers]
        return str(sum(numbers))

    # Function Combine
    def combine(list):
        ALL = {'zero': 0,
               'one': 1, 'a': 1, 'an': 1,
               'two': 2,
               'three': 3,
               'four': 4,
               'five': 5,
               'six': 6,
               'seven': 7,
               'eight': 8,
               'nine': 9,
               'ten': 10,
               'eleven': 11,
               'twelve': 12,
               'thirteen': 13,
               'fourteen': 14,
               'fifteen': 15,
               'sixteen': 16,
               'seventeen': 17,
               'eighteen': 18,
               'nineteen': 19,
               'twenty': 20,
               'thirty': 30,
               'forty': 40,
               'fifty': 50,
               'sixty': 60,
               'seventy': 70,
               'eighty': 80,
               'ninety': 90,
               'hundred': 100,
               'thousand': 1000,
               'million': 1000000,
               'billion': 1000000000,
               }
        pointer = []
        for s in list:
            if s.lower() in ALL:
                pointer.append(1)
            else:
                pointer.append(0)

        result = []
        for i in range(len(list)):
            result.append(list[i])
            if i > 0:
                if pointer[i] == 1 and pointer[i - 1] == 1:
                    temp = result[-2] + ' ' + result[-1]
                    result[-2] = temp
                    result.pop()

        return result

    def untokenize(words):
        text = ' '.join(words)
        step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
            "can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        temp = step6
        count = 0
        while ' " ' in temp:
            if count % 2 == 0:
                temp = re.sub(' " ', ' "', temp, 1)
            if count % 2 == 1:
                temp = re.sub(' " ', '" ', temp, 1)
            count += 1
        step7 = temp
        last_position = step7.rfind(' "')
        step7 = step7[:last_position] + step7[step7.rfind('"'):]

        return step7.strip()



    # Normalize
    result = []
    s1 = tokenize_regex(text)
    s2 = combine(s1)

    for seg in s2:
        temp = seg
        if parse_int(seg) != '0':
            temp = parse_int(seg)
        elif seg == 'zero':
            temp = '0'
        result.append(temp)

    FRC = ['half','third','fourth','fifth','sixth','seventh','ninth','tenth']
    for i, s in enumerate(result):

        # Getting rid of ordinals
        if s in FRC and i > 0:
            result[i-1] = s2[i-1]

        # Getting rid of ordinals
        if s == '1' and (s2[i].lower() == "a" or s2[i].lower() == "an"):
            result[i] = s2[i]

        # Getting rid of ordinals
        if s == 'point' and i > 0 and (result[i - 1].isnumeric() and result[i + 1].isnumeric()):
            result[i - 1] = s2[i - 1]
            result[i + 1] = s2[i + 1]
    # Untokenize
    return untokenize(result)


def normalize_extra(text):

    RE_TOK = re.compile(r'([",.]|n\'t|\'|\?|\!|\s+)')

    def tokenize_regex(text):
        prev_idx = 0
        tokens = []
        for m in RE_TOK.finditer(text):
            t = text[prev_idx:m.start()].strip()
            if t: tokens.append(t)
            t = m.group().strip()
            if t:
                if tokens and tokens[-1] in {'Mr', 'Ms'} and t == '.':
                    tokens[-1] = tokens[-1] + t
                else:
                    tokens.append(t)
            prev_idx = m.end()

        t = text[prev_idx:]
        if t:  tokens.append(t)
        return tokens

    # Function ParseInt
    def parse_int(string):
        NUM = {'zero': 0,
               'one': 1, 'a': 1, 'an': 1,
               'two': 2,
               'three': 3,
               'four': 4,
               'five': 5,
               'six': 6,
               'seven': 7,
               'eight': 8,
               'nine': 9,
               'ten': 10,
               'eleven': 11,
               'twelve': 12,
               'thirteen': 13,
               'fourteen': 14,
               'fifteen': 15,
               'sixteen': 16,
               'seventeen': 17,
               'eighteen': 18,
               'nineteen': 19,
               'twenty': 20,
               'thirty': 30,
               'forty': 40,
               'fifty': 50,
               'sixty': 60,
               'seventy': 70,
               'eighty': 80,
               'ninety': 90,

               }
        numbers = []
        for token in string.replace('-', ' ').split(' '):
            if token.lower() in NUM:
                numbers.append(NUM[token.lower()])
            elif token.lower() == 'hundred':
                # numbers[-1] *= 100
                numbers = [x * 100 for x in numbers]
            elif token.lower() == 'thousand':
                numbers = [x * 1000 for x in numbers]
            elif token.lower() == 'million':
                numbers = [x * 1000000 for x in numbers]
            elif token.lower() == 'billion':
                numbers = [x * 1000000000 for x in numbers]
        return str(sum(numbers))

    # Function Combine
    def combine(list):
        ALL = {'zero': 0,
               'one': 1, 'a': 1, 'an': 1,
               'two': 2,
               'three': 3,
               'four': 4,
               'five': 5,
               'six': 6,
               'seven': 7,
               'eight': 8,
               'nine': 9,
               'ten': 10,
               'eleven': 11,
               'twelve': 12,
               'thirteen': 13,
               'fourteen': 14,
               'fifteen': 15,
               'sixteen': 16,
               'seventeen': 17,
               'eighteen': 18,
               'nineteen': 19,
               'twenty': 20,
               'thirty': 30,
               'forty': 40,
               'fifty': 50,
               'sixty': 60,
               'seventy': 70,
               'eighty': 80,
               'ninety': 90,
               'hundred': 100,
               'thousand': 1000,
               'million': 1000000,
               'billion': 1000000000,
               }
        pointer = []
        for s in list:
            if s.lower() in ALL:
                pointer.append(1)
            else:
                pointer.append(0)

        result = []
        for i in range(len(list)):
            result.append(list[i])
            if i > 0:
                if pointer[i] == 1 and pointer[i - 1] == 1:
                    temp = result[-2] + ' ' + result[-1]
                    result[-2] = temp
                    result.pop()

        return result

    def untokenize(words):
        text = ' '.join(words)
        step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
            "can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        temp = step6
        count = 0
        while ' " ' in temp:
            if count % 2 == 0:
                temp = re.sub(' " ', ' "', temp, 1)
            if count % 2 == 1:
                temp = re.sub(' " ', '" ', temp, 1)
            count += 1
        step7 = temp
        last_position = step7.rfind(' "')
        step7 = step7[:last_position] + step7[step7.rfind('"'):]

        return step7.strip()

    # Normalize
    result = []
    s1 = tokenize_regex(text)
    s2 = combine(s1)

    for seg in s2:
        temp = seg
        if parse_int(seg) != '0':
            temp = parse_int(seg)
        elif seg == 'zero':
            temp = '0'
        result.append(temp)

    FRC = ['half', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'ninth', 'tenth']
    for i, s in enumerate(result):

        # Getting rid of ordinals
        if s in FRC and i > 0:
            result[i - 1] = s2[i - 1]

        # Getting rid of indefinite articles
        #if s == '1' and (s2[i].lower() == "a" or s2[i].lower() == "an"):
        #    result[i] = s2[i]

        # Getting rid of fractions
        if s == 'point' and i > 0 and (result[i - 1].isnumeric() and result[i + 1].isnumeric()):
            result[i - 1] = s2[i - 1]
            result[i + 1] = s2[i + 1]

    return untokenize(result)


if __name__ == '__main__':
    S = [
        'I met twelve people',
        'I have one brother and two sisters',
        'A year has three hundred sixty five days',
        'I made a million dollars'
    ]

    T = [
        'I met 12 people',
        'I have 1 brother and 2 sisters',
        'A year has 365 days',
        'I made 1000000 dollars'
    ]

    correct = 0
    for s, t in zip(S, T):
        if normalize(s) == t:
            correct += 1

    print('Score: {}/{}'.format(correct, len(S)))