# -*- coding: utf-8 -*-
# @Time : 2023/2/5
# @Author : YEY
# @File : naive_holdem.py

from collections import Counter
from collections import defaultdict
from enum import Enum, unique
from itertools import product
import matplotlib.pyplot as plt
import random
from utils.my_decorators import dividing_line
from utils.my_decorators import plt_support_cn


@unique
class Suit(Enum):
    SPADE = '黑桃'
    HEART = '红桃'
    CLUB = '梅花'
    DIAMOND = '方片'
    MONO = '黑白'
    COLOR = '彩色'

    @property
    def face(self):
        face_dict = {'SPADE': '\u2660', 'HEART': '\u2665', 'CLUB': '\u2666', 'DIAMOND': '\u2663', 'MONO': '小',
                     'COLOR': '大'}
        return face_dict[self.name]


@unique
class Number(Enum):
    TWO = '2', 2
    THREE = '3', 3
    FOUR = '4', 4
    FIVE = '5', 5
    SIX = '6', 6
    SEVEN = '7', 7
    EIGHT = '8', 8
    NINE = '9', 9
    TEN = '10', 10
    JACK = 'J', 11
    QUEEN = 'Q', 12
    KING = 'K', 13
    ACE = 'A', 14
    MONO_JOKER = 'Joker', 50
    COLOR_JOKER = 'JOKER', 100

    @property
    def face(self):
        return self.value[0]

    @property
    def number_value(self):
        if self == Number.MONO_JOKER or self == Number.COLOR_JOKER:
            val = tuple(range(1, 15))
        elif self == Number.ACE:
            val = (self.value[1], 1)
        else:
            val = (self.value[1],)
        return val


class Card:
    def __init__(self, suit, number):
        assert Card._check_args(suit, number), "请使用合法的花色与点数组合。"
        self.suit = suit
        self.number = number
        self.face = suit.face + number.face
        self.number_value = number.number_value

    def __eq__(self, other):
        return max(self.number_value) == max(other.number_value)

    def __gt__(self, other):
        return max(self.number_value) > max(other.number_value)

    def __lt__(self, other):
        return max(self.number_value) < max(other.number_value)

    def __str__(self):
        return "【单牌】{}: 花色为{}, 点数为{}".format(self.face, self.suit.value, self.number.face)

    @classmethod
    def _check_args(cls, suit, number):
        if not (isinstance(suit, Suit) and isinstance(number, Number)):
            return False
        elif suit in (Suit.MONO, Suit.COLOR) and number not in (Number.MONO_JOKER, Number.COLOR_JOKER):
            return False
        elif suit not in (Suit.MONO, Suit.COLOR) and number in (Number.MONO_JOKER, Number.COLOR_JOKER):
            return False
        else:
            return True


@unique
class HandPattern(Enum):
    ROYAL_STRAIGHT_FLUSH = '皇家同花顺', 10
    STRAIGHT_FLUSH = '同花顺', 9
    FOUR_OF_A_KIND = '铁支', 8
    FULL_HOUSE = '葫芦', 7
    FLUSH = '同花', 6
    STRAIGHT = '顺子', 5
    THREE_OF_A_KIND = '三条', 4
    TWO_PAIR = '两对', 3
    PAIR = '对子', 2
    HIGH_CARD = '散牌', 1

    @property
    def pattern_name(self):
        return self.value[0]

    @property
    def pattern_value(self):
        return self.value[1]


class Hand:
    def __init__(self, cards):
        self.cards = cards
        self.face = None
        self.sorted_face = None
        self.pattern = None
        self.value = None
        self._all_combinations = None
        self._all_comb_counts = None
        self._get_all_combination()
        self._get_all_comb_count()
        self._compute_pattern_and_value()
        self._hand_sort()

    def __eq__(self, other):
        return self.value == other.value

    def __gt__(self, other):
        return self.value > other.value

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        return "【手牌】{}: 牌型为{}, 分值为{}".format('、'.join(self.sorted_face), self.pattern.pattern_name, self.value)

    def _get_all_combination(self):
        possible_numbers = [c.number_value for c in self.cards]
        all_combinations = list(product(*possible_numbers))
        self._all_combinations = [tuple(sorted(comb)) for comb in all_combinations]

    def _get_all_comb_count(self):
        self._all_comb_counts = set([tuple(sorted(Counter(c).values(), reverse=True)) for c in self._all_combinations])

    def _is_royal_straight_flush(self):
        max_num_val = max(max(n.number_value for n in Number))
        royal_straight = tuple(range(max_num_val - len(self.cards) + 1, max_num_val + 1))
        return self._is_straight_flush() and royal_straight in self._all_combinations

    def _is_straight_flush(self):
        return self._is_straight() and self._is_flush()

    def _is_four_of_a_kind(self):
        return (4, 1) in self._all_comb_counts

    def _is_full_house(self):
        return (3, 2) in self._all_comb_counts

    def _is_flush(self):
        return len(set(c.suit for c in self.cards if c.suit not in (Suit.MONO, Suit.COLOR))) == 1

    def _is_straight(self):
        def check_continuity(nums):
            sorted_nums = sorted(nums)
            return list(range(min(sorted_nums), max(sorted_nums) + 1)) == sorted_nums

        possible_numbers = [c.number_value for c in self.cards]
        all_combination = list(product(*possible_numbers))

        return any(check_continuity(comb) for comb in all_combination)

    def _is_three_of_a_kind(self):
        return (3, 1, 1) in self._all_comb_counts

    def _is_two_pair(self):
        return (2, 2, 1) in self._all_comb_counts

    def _is_pair(self):
        return (2, 1, 1, 1) in self._all_comb_counts

    def _compute_pattern_and_value(self):
        # 判断牌型
        if self._is_royal_straight_flush():
            self.pattern = HandPattern.ROYAL_STRAIGHT_FLUSH
        elif self._is_straight_flush():
            self.pattern = HandPattern.STRAIGHT_FLUSH
        elif self._is_four_of_a_kind():
            self.pattern = HandPattern.FOUR_OF_A_KIND
        elif self._is_full_house():
            self.pattern = HandPattern.FULL_HOUSE
        elif self._is_flush():
            self.pattern = HandPattern.FLUSH
        elif self._is_straight():
            self.pattern = HandPattern.STRAIGHT
        elif self._is_three_of_a_kind():
            self.pattern = HandPattern.THREE_OF_A_KIND
        elif self._is_two_pair():
            self.pattern = HandPattern.TWO_PAIR
        elif self._is_pair():
            self.pattern = HandPattern.PAIR
        else:
            self.pattern = HandPattern.HIGH_CARD
        # 匹配牌型对应分值
        self.value = self.pattern.pattern_value

    def _hand_sort(self):
        self.face = [c.face for c in self.cards]
        suit_number_cards = [(c.face, c.number.value[1]) for c in self.cards]
        self.sorted_face = [c[0] for c in sorted(suit_number_cards, key=lambda x: x[1], reverse=False)]


class HoldEmGame:
    def __init__(self, card_num=5, player_num=3, deck_num=1, with_joker=False, round_time=10000):
        self.card_num = card_num
        self.player_num = player_num
        self.deck_num = deck_num
        self.with_joker = with_joker
        self.round_time = round_time
        self.one_deck_cards = None
        self.all_cards = None
        self.round_hands = None
        self._get_all_cards()
        self.distribute_hand()

    def _get_one_deck_cards(self):
        suits = [s for s in Suit if s not in (Suit.MONO, Suit.COLOR)]
        numbers = [n for n in Number if n not in (Number.MONO_JOKER, Number.COLOR_JOKER)]
        normal_cards = [Card(s, n) for s, n in [*product(suits, numbers)]]
        if not self.with_joker:
            self.one_deck_cards = normal_cards
        else:
            joker_cards = [Card(Suit.MONO, Number.MONO_JOKER), Card(Suit.COLOR, Number.COLOR_JOKER)]
            self.one_deck_cards = normal_cards + joker_cards

    def _get_all_cards(self):
        self._get_one_deck_cards()
        if self.card_num * self.player_num > len(self.one_deck_cards) * self.deck_num:
            self.deck_num = self.card_num * self.player_num // len(self.one_deck_cards) + 1
        self.all_cards = self.one_deck_cards * self.deck_num

    def distribute_hand(self):
        random.shuffle(self.all_cards)
        self.round_hands = [Hand(self.all_cards[i:i + self.card_num]) for i in
                            range(0, self.player_num * self.card_num, self.card_num)]

    def get_pattern_count(self):
        hand_pattern_counter = Counter()
        total_hand_count = 0
        for _ in range(self.round_time):
            self.distribute_hand()
            for hand in self.round_hands:
                total_hand_count += 1
                hand_pattern_counter[hand.pattern.pattern_name] += 1
        return hand_pattern_counter, total_hand_count


class Task:

    @staticmethod
    def get_compare_result(obj1, obj2):
        if obj1 == obj2:
            return '='
        elif obj1 > obj2:
            return '>'
        else:
            return '<'

    @staticmethod
    @dividing_line('单牌比较大小测试')
    def one_card_compile_test():
        card_1 = Card(Suit.SPADE, Number.ACE)
        card_2 = Card(Suit.SPADE, Number.QUEEN)
        result = Task.get_compare_result(card_1, card_2)
        print('【单牌1】: {}\n【单牌2】: {}\n【比较结果】: 单牌1 {} 单牌2'.format(card_1, card_2, result))

    @staticmethod
    @dividing_line('手牌比较大小测试')
    def one_hand_compile_test():
        hand_1 = Hand([Card(Suit.SPADE, Number.ACE),
                       Card(Suit.SPADE, Number.QUEEN),
                       Card(Suit.SPADE, Number.TEN),
                       Card(Suit.SPADE, Number.JACK),
                       Card(Suit.COLOR, Number.COLOR_JOKER)])

        hand_2 = Hand([Card(Suit.SPADE, Number.ACE),
                       Card(Suit.MONO, Number.MONO_JOKER),
                       Card(Suit.HEART, Number.ACE),
                       Card(Suit.CLUB, Number.SEVEN),
                       Card(Suit.COLOR, Number.COLOR_JOKER)])

        result = Task.get_compare_result(hand_1, hand_2)
        print('【手牌1】: {}\n【手牌2】: {}\n【比较结果】: 手牌1 {} 手牌2'.format(hand_1, hand_2, result))

    @staticmethod
    @plt_support_cn
    @dividing_line('不同牌型概率计算')
    def compute_pattern_probs(card_num=5, player_num=3, deck_num=1, with_joker=False, round_time=10000):
        pattern_probs_dict = defaultdict(float)
        holdem_game = HoldEmGame(card_num=card_num, player_num=player_num, deck_num=deck_num, with_joker=with_joker,
                                 round_time=round_time)
        pattern_counter, total_hand = holdem_game.get_pattern_count()
        for k in pattern_counter.keys():
            pattern_probs_dict[k] = pattern_counter[k] / total_hand

        patterns = [p.pattern_name for p in HandPattern]
        probs = [pattern_probs_dict[p] for p in patterns]
        for pattern, prob in zip(patterns, probs):
            print('【{}】: {}'.format(pattern, prob))

        fig_name = '各牌型概率'
        plt.figure(figsize=(10, 6))
        plt.title(fig_name)
        plt.ylim(0, 1)
        plt.bar(patterns, probs)
        for pattern, prob in zip(patterns, probs):
            plt.text(pattern, prob, prob, ha='center', va='bottom')
        plt.savefig(fig_name, dpi=600)
        plt.show()

        return pattern_probs_dict


if __name__ == '__main__':
    Task.one_card_compile_test()
    Task.one_hand_compile_test()

    random.seed(1)
    CARD_NUM = 5
    PLAYER_NUM = 10
    DECK_NUM = 1
    WITH_JOKER = True
    ROUND_TIME = 100000
    pattern_probs = Task.compute_pattern_probs(card_num=CARD_NUM, player_num=PLAYER_NUM, deck_num=DECK_NUM,
                                               with_joker=WITH_JOKER, round_time=ROUND_TIME)
