import regex as re
from vncorenlp import VnCoreNLP

from utils import load_yaml

PATHS = load_yaml("paths")
with open(PATHS["stopwords"], "r") as f:
    stopwords = f.read().split("\n")

rdrsegmenter = VnCoreNLP(
    PATHS["vncoreNLP"], annotators="wseg", max_heap_size="-Xmx500m"
)


def vncore_tokenizer(text):
    words = rdrsegmenter.tokenize(text)
    text = " ".join([" ".join(x) for x in words])
    return text


class NLP(object):
    uniChars = (
        "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
        "ÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    )

    bang_nguyen_am = [
        ["a", "à", "á", "ả", "ã", "ạ", "a"],
        ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
        ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
        ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
        ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
        ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
        ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
        ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "oo"],
        ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
        ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
        ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
        ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
    ]

    def __init__(self, tokenize_method=vncore_tokenizer) -> None:
        self.tokenize_method = tokenize_method
        self.dicchar = self._loaddicchar()
        self.nguyen_am_to_ids = self._nguyen_am_to_ids()

    def _loaddicchar(self):
        dic = {}
        char1252 = (
            "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|"
            "ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|"
            "Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
                "|"
            )
        )
        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ' \
                   '|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|' \
                   'Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
            "|"
        )
        for i in range(len(char1252)):
            dic[char1252[i]] = charutf8[i]
        return dic

    def _nguyen_am_to_ids(self) -> dict:
        nguyen_am_to_ids = {}

        for i in range(len(self.bang_nguyen_am)):
            for j in range(len(self.bang_nguyen_am[i]) - 1):
                nguyen_am_to_ids[self.bang_nguyen_am[i][j]] = (i, j)

        return nguyen_am_to_ids

    def convert_unicode(self, txt):
        # Hàm chuyển Unicode dựng sẵn về Unicde tổ hợp (phổ biến hơn)
        return re.sub(
            r"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|"
            "ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|"
            "Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ",
            lambda x: self.dicchar[x.group()],
            txt,
        )

    def chuan_hoa_dau_tu_tieng_viet(self, word):
        if not self.is_valid_vietnam_word(word):
            return word

        chars = list(word)
        dau_cau = 0
        nguyen_am_index = []
        qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
            if x == -1:
                continue
            elif x == 9:  # check qu
                if index != 0 and chars[index - 1] == "q":
                    chars[index] = "u"
                    qu_or_gi = True
            elif x == 5:  # check gi
                if index != 0 and chars[index - 1] == "g":
                    chars[index] = "i"
                    qu_or_gi = True
            if y != 0:
                dau_cau = y
                chars[index] = self.bang_nguyen_am[x][0]
            if not qu_or_gi or index != 1:
                nguyen_am_index.append(index)
        if len(nguyen_am_index) < 2:
            if qu_or_gi:
                if len(chars) == 2:
                    x, y = self.nguyen_am_to_ids.get(chars[1], (-1, None))
                    chars[1] = self.bang_nguyen_am[x][dau_cau]
                else:
                    x, y = self.nguyen_am_to_ids.get(chars[2], (-1, -1))
                    if x != -1:
                        chars[2] = self.bang_nguyen_am[x][dau_cau]
                    else:
                        chars[1] = (
                            self.bang_nguyen_am[5][dau_cau]
                            if chars[1] == "i"
                            else self.bang_nguyen_am[9][dau_cau]
                        )
                return "".join(chars)
            return word

        for index in nguyen_am_index:
            x, y = self.nguyen_am_to_ids[chars[index]]
            if x == 4 or x == 8:  # ê, ơ
                chars[index] = self.bang_nguyen_am[x][dau_cau]
                return "".join(chars)

        if len(nguyen_am_index) == 2:
            if nguyen_am_index[-1] == len(chars) - 1:
                x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[0]]]
                chars[nguyen_am_index[0]] = self.bang_nguyen_am[x][dau_cau]
            else:
                x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[1]]]
                chars[nguyen_am_index[1]] = self.bang_nguyen_am[x][dau_cau]
        else:
            x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = self.bang_nguyen_am[x][dau_cau]
        return "".join(chars)

    def is_valid_vietnam_word(self, word):
        chars = list(word)
        nguyen_am_index = -1
        for index, char in enumerate(chars):
            x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
            if x != -1:
                if nguyen_am_index == -1:
                    nguyen_am_index = index
                else:
                    if index - nguyen_am_index != 1:
                        return False
                    nguyen_am_index = index
        return True

    def chuan_hoa_dau_cau_tieng_viet(self, sentence):
        """
        Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
        :param sentence:
        :return:
        """
        sentence = sentence.lower()
        words = sentence.split()
        for index, word in enumerate(words):
            cw = re.sub(r"(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)", r"\1/\2/\3", word).split(
                "/"
            )
            if len(cw) == 3:
                cw[1] = self.chuan_hoa_dau_tu_tieng_viet(cw[1])
            words[index] = "".join(cw)
        return " ".join(words)

    def remove_html(self, txt):
        return re.sub(r"<[^>]*>", "", txt)

    def preprocess_text(self, document):
        document = self.remove_html(document)
        document = self.convert_unicode(document)
        document = self.chuan_hoa_dau_cau_tieng_viet(document)
        document = document.lower()
        document = self.tokenize_method(document)
        # xóa các ký tự không cần thiết
        document = re.sub(
            r"[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]",
            " ",
            document,
        )
        # xóa khoảng trắng thừa
        document = re.sub(r"\s+", " ", document).strip()
        return document


def remove_stopwords(document, stopwords):
    words = []
    for word in document.strip().split():
        if word not in stopwords:
            words.append(word)
    return " ".join(words)


def remove_number(document):
    return " ".join([word for word in document.split(" ") if not word.isdigit()])


if __name__ == "__main__":
    from pyvi.ViTokenizer import tokenize

    s = "chúng ta phải sống thật hạnh phúc"
    s2 = "Một Granada xếp áp chót bảng tất nhiên không thể là mối hiểm họa cho Real Madrid (R.M), nhưng HLV Zinedine Zidane có lý do e ngại cho cuộc đấu này khi ông thiếu vắng hàng loạt trụ cột vì những lý do khác nhau.\nCụ thể ở trận này Gareth Bale, Pepe, Mateo Kovacic và Lucas Vazquez không thể ra sân vì chưa kịp hồi phục chấn thương, ngoài ra đội trưởng Sergio Ramos cũng vắng mặt vì án treo giò. Trong tình cảnh đó, Ronaldo tất nhiên gánh vác toàn bộ sự kỳ vọng của CĐV R.M.\nTrong trận đấu đầu năm của R.M đại thắng Sevilla 3-0 ở Cúp nhà vua cách đây 3 ngày, Ronaldo được HLV Zidane cho nghỉ ngơi dưỡng sức. Vì vậy, Ronaldo hoàn toàn sung sức và đặc biệt háo hức cho trận đấu này. Năm 2016 là một năm quá mỹ mãn, thậm chí có thể xem là thành công nhất trong sự nghiệp của Ronaldo. Dù vậy, siêu sao người Bồ Đào Nha đã tỏ dấu hiệu sa sút vì tuổi tác trong nửa đầu mùa giải 2016-2017. Từ đầu mùa đến giờ, Ronaldo mới ghi vỏn vẹn 16 bàn trên mọi mặt trận, đạt hiệu suất làm bàn chỉ bằng khoảng một nửa so với những mùa giải trước đó.\nSự sa sút đó không có gì khó hiểu, bởi tháng 2 tới Ronaldo sẽ tròn 32 tuổi. Việc Pepe không thể ra sân đêm mai đưa Ronaldo đến một “vinh dự” đặc biệt: cầu thủ già nhất của R.M trên sân. Nếu đội phó Marcelo ra sân, dù băng đội trưởng vẫn chưa thể đến tay của Ronaldo, nhưng anh vẫn có thể được xem như một thủ lĩnh thực thụ của R.M lúc này.\nNgoài khao khát của Ronaldo, toàn đội R.M còn hướng đến một cột mốc đặc biệt - cân bằng kỷ lục 39 trận bất bại liên tiếp  của Barca.\nViệc thiết lập kỷ lục mới hoàn toàn nằm trong tầm tay của R.M bởi sau Granada, “kền kền trắng” chỉ phải đá trận lượt về Cúp nhà vua với một Sevilla đã hết hi vọng (thua 0-3 ở lượt đi). Trong trận lượt đi, những cầu thủ dự bị của R.M như James Rodriguez, Alvaro Moratam, Nacho đã thi đấu rất tốt, khiến HLV Zidane hoàn toàn có thể yên tâm chuyện nhiều trụ cột phải vắng mặt."
    nl = NLP()
    print(nl.preprocess_text(s2))
