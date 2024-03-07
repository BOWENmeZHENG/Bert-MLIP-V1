import torch
import json
import math

def text2token(tokenizer, text):
    text_list = tokenizer.tokenize(text)
    token_list = tokenizer.convert_tokens_to_ids(text_list)
    token = torch.tensor(token_list)[None, :]
    return token

def list2token(tokenizer, text_list, max_length):
    token_list = tokenizer.convert_tokens_to_ids(text_list)
    token_list_padded = token_list + [0] * (max_length - len(token_list))
    token = torch.tensor(token_list_padded)[None, :]
    return token

def read_data(json_file, max_length):
    data_list = []
    with open(json_file) as f:
        for jsonObj in f:
            record = json.loads(jsonObj)
            if len(record['words']) > max_length:
                record['words'] = record['words'][:max_length]
                record['ner'] = record['ner'][:max_length]
            data_list.append(record)
    return data_list

def cat2digit(classes, cat_text, max_length):
    label_digit = [classes.get(item, item) for item in cat_text]
    label_digit_padded = label_digit + [len(classes)] * (max_length - len(label_digit))
    att_mask = [1] * len(label_digit) + [0] * (max_length - len(label_digit))
    return torch.tensor(label_digit_padded), torch.tensor(att_mask)

def to_batches(x, batch_size):
    num_batches = math.ceil(x.size()[0] / batch_size)
    return [x[batch_size * y: batch_size * (y+1),:] for y in range(num_batches)]

def accuracy(index_other, index_pad, y_pred, y):
    indices = ((index_other < y) & (y < index_pad)).nonzero(as_tuple=True)  # words with entity
    _, predicted_classes = y_pred[indices[0], :, indices[1]].max(dim=1)
    true_classes = y[indices[0], indices[1]]
    accuracy = torch.eq(predicted_classes, true_classes).sum() / true_classes.shape[0]
    return accuracy, predicted_classes, true_classes
