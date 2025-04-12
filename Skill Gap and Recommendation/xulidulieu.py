import re

def Pre(df):
    # Lấy tên cột cuối cùng
    last_column = df.columns[-1]
    
    # Hàm xử lý văn bản
    def clean_text(text):
        # Chuyển thành chữ thường
        text = text.lower()
        # Thay thế các ký tự đặc biệt bằng khoảng trắng

        text = text.replace("'", " ")
        text = text.replace('"', " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace("[", " ")
        text = text.replace("]", " ")
        text = text.replace("{", " ")
        text = text.replace("}", " ")
        text = text.replace("!", " ")
        text = text.replace("?", " ")
        text = text.replace(":", " ")
        text = text.replace(";", " ")
        text = text.replace(",", " ")
        text = text.replace(".", " ")
        text = text.replace("-", " ")
        text = text.replace("_", " ")
        text = text.replace("+", " ")
        text = text.replace("=", " ")
        text = text.replace("*", " ")
        text = text.replace("&", " ")
        text = text.replace("^", " ")
        text = text.replace("%", " ")
        text = text.replace("$", " ")
        text = text.replace("#", " ")
        text = text.replace("@", " ")
        text = text.replace("~", " ")
        text = text.replace("`", " ")
        text = text.replace("|", " ")
        text = text.replace("<", " ")
        text = text.replace(">", " ")
        text = text.replace("/", " ")
        text = text.replace("\\", " ")
        
       
        return text
    # Áp dụng xử lý văn bản cho cột cuối cùng
    df[last_column] = df[last_column].apply(clean_text)
    return df
