from nltk.tokenize import word_tokenize
from nltk import pos_tag

def classify_query(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    if not pos_tags:  
        return "Other"

    for word, tag in pos_tags:
        word_lower = word.lower()
        
        if tag in ["WP", "WP$", "WRB"]:  
            if word_lower in ["what"]:
                return "What"
            elif word_lower in ["which"]:
                return "Which"
            elif word_lower in ["how"]:
                return "How"
            elif word_lower in ["who", "whom", "whose"]:
                return "Who"
            elif word_lower in ["where"]:
                return "Where"
            elif word_lower in ["when"]:
                return "When"
            elif word_lower in ["why"]:
                return "Why"
    
    first_word = pos_tags[0][0].lower()
    if first_word in ["is", "are", "does", "do", "did", "can", "will", "should"]:
        return "YesNo"
    
    return "Other"