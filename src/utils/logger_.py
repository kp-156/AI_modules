import sys
import traceback
import json

def execute_and_log(func, *args, **kwargs):
    result = {'success': False, 'result': None, 'output': None}
    try:
        # result['input'] = kwargs
        result['output'] = func(*args, **kwargs) # { }
        result['success'] = True
        result['result'] = result['output']
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        # result['input'] = kwargs
        result['result'] = {
            'exception_type': str(exc_type),
            'exception_value': str(exc_value),
            'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback)
        }
    return result

def replace_keys_in_nested_dict(dict_, old_char, new_char):
    new_dict = {}
    for key, value in dict_.items():
        new_key = key.replace(old_char, new_char)
        if isinstance(value, dict):
            new_value = replace_keys_in_nested_dict(value, old_char, new_char)
        else:
            new_value = value
        new_dict[new_key] = new_value
    return new_dict
