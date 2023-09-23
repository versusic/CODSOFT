import nltk
from nltk.chat.util import Chat, reflections

rules = [
    [r"my name is (.*)", ["hello %1, how are you today?"],],
    [r"hi|hey|hello", ["hello, how are you?", "hi there, how are you?"],],
    [r"I am fine|I am okay|I am good", ["nice to hear that, how can I help you?", "that's good, how can I help you?"],],
    [r"what can you do?|What do you do?", ["I can answer some basic chat questions"],],
    [r"how are you?|How’s it going?|How’s your day going?", ["I am a bot, I don't feel bad or good"],],
    [r"what is your name?", ["I don't have a name"],],
    [r"what are you?", ["I am a rule-based chatbot, I am happy to help you"],],
    [r"how can you help me?", ["I can be assigned to answer common questions in your field, but you have to program me "
                               "with what you need"],],
    [r"Where do you live?|Where are you from?", [""],],
    [r"nice|good", ["thanks"],],
    [r"like what?", ["I can tell you how to make a simple rule-based chatbot like me"], ],
    [r"how?", ["use nltk library, and import Chat and reflection, then define your rules, you can search more on "
               "internet to build a better one"], ],
    [r"thanks", ["you are welcome"], ],
    [r"bye", ["see you soon"], ],
    [r"", ["I don't know", "I don't understand", "I wasn't programmed to answer this"], ],
         ]

def chatbot():
    print("hi, I am a rule-based chatbot")

chat = Chat(rules, reflections)
chatbot()
chat.converse()
if __name__ == "__main__":
    chatbot()