
from typing import Union, Callable, Any
import random
from tensorflow.keras.losses import Loss


class ParamsGenerator:

    def __init_dict(self) -> None:
        self.d = {
        'embedding_params': {
            'output_dim': 124,
            'mask_zero': True},

        'recurrent_list': [
            None,
            {'units': 64,
             'activation': 'linear',
             'return_sequences': True,
             'dropout': 0.0,
             'recurrent_dropout': 0.0}],

        'use_attention': True,

        'optimizer_list': [
            'adam',
            {'lr': 0.001,
             'beta_1': 0.9,
             'beta_2': 0.999,
             'amsgrad': False,
             'clipnorm': 1.0}
        ],

        'loss_function': None,

        'model_name': None

    }

    @staticmethod
    def __get_function(element : Any, new_val : Any):
        if new_val is not None:
            current_element = new_val
        else:
            current_element = element

        if type(current_element) == set:
            return function_wrapper(random.choice, seq=list(current_element))
        elif type(current_element) == list:
            if len(current_element) == 2:
                if type(element) == int:
                    f = random.randint
                elif type(element) == float:
                    f = random.uniform
                else:
                    print(type(element), element, current_element)
                return function_wrapper(f, a=current_element[0], b=current_element[1])
        else:
            return function_wrapper(None, a=current_element)

    def __process_element(self, key : str, inner_key2:int, inner_key: str,  element : Any, new_val : Any = None) -> None:
        if inner_key is None and inner_key2 is None:
            self.d[key] = self.__get_function(element, new_val)
        elif inner_key is None:
            self.d[key][inner_key2] = self.__get_function(element, new_val)
        elif inner_key2 is None:
            self.d[key][inner_key] =self.__get_function(element, new_val)
        else:
            self.d[key][inner_key2][inner_key] = self.__get_function(element, new_val)

    def __init__(self, model_name : str,
                 recurrent_type: Union[str, set] = 'lstm',
                 loss_function: Union[str, set, Loss] = 'binary_crossentropy',
                 **kwargs):
        self.__init_dict()
        self.d['model_name'] = model_name
        self.d['recurrent_list'][0] = recurrent_type
        self.d['loss_function'] = loss_function

        for key in self.d:
            element = self.d[key]
            inner_key2 = None
            if type(element) == list:
                if key in kwargs.keys():
                    new_val = self.d[key][0]
                else:
                    new_val = None

                self.__process_element(key, 0, None, element[0], new_val)
                element = element[1]
                inner_key2 = 1
            if type(element) == dict:
                for inner_key in element:
                    if inner_key in kwargs.keys():
                        new_val = kwargs[inner_key]
                    else:
                        new_val = None
                    self.__process_element(key, inner_key2, inner_key, element[inner_key], new_val)

            if type(element) != dict and type(element) != list:
                if key in kwargs.keys():
                    new_val = kwargs[key]
                else:
                    new_val = None
                self.__process_element(key, None, None, element, new_val)

    def generate(self, num_instances : int) -> list:
        to_return = list()
        for _ in range(num_instances):
            new_d = self.d.copy()
            for key in new_d:
                if type(new_d[key]) == list:
                    new_d[key] = self.d[key].copy()
                    new_d[key][1] = self.d[key][1].copy()
                elif type(new_d[key]) == dict:
                    new_d[key] = self.d[key].copy()
            for key in new_d:
                element = new_d[key]
                if type(element) == list:
                    element[0] = element[0]()
                    element = element[1]
                if type(element) == dict:
                    for inner_k in element:
                        element[inner_k] = element[inner_k]()
                if type(element) != dict and type(element) != list:
                    new_d[key] = new_d[key]()

            to_return.append(new_d)
        return to_return


def function_wrapper(func, **kwargs) -> Callable:
    def f():
        if func is not None:
            return func(**kwargs)
        else:
            for k in kwargs:
                return kwargs[k]
    return f


random.seed(43)




