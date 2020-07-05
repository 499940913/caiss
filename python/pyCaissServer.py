import json

import numpy
import tornado.ioloop
import tornado.web
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer

from python.pyCaiss import *

LIB_PATH = r'/Users/chunel/Documents/code/cpp/caiss/python/libCaiss.dylib'
WORD_MODEL_PATH = r'/Users/chunel/Documents/code/cpp/models/bert_71290words_768dim.caiss'
SENT_MODEL_PATH = r'/Users/chunel/Documents/code/cpp/models/bert-valid-38550sent-768dim.caiss'

MAX_THREAD_SIZE = 1
DIM = 768

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, welcome to the world of Caiss")

class CaissWordHandler(tornado.web.RequestHandler):
    def get(self):
        query_word = self.get_argument('query', '')
        if len(query_word) == 0:
            self.write('please enter query word.')
            return
        top_k = self.get_argument('top', '5')

        ret, result_str = caiss.sync_search(handle, query_word, CAISS_SEARCH_WORD, int(top_k), 0)
        if 2 == ret:
            self.write('this is not a word : [' + query_word + ']')
            return
        elif 0 != ret:
            self.write('search failed for the reason of : ' + ret)
            return

        result_dict = json.loads(result_str)
        word_list = list()
        for info in result_dict['details']:
            word_list.append(info['label'])

        self.write('the query word is [' + query_word + '].')
        self.write('<br>')
        self.write('the word you also want to know maybe : ')
        self.write(str(word_list))
        self.write('.<br>')

class CaissSentenceHandler(tornado.web.RequestHandler):
    def get(self):
        query_sent = self.get_argument('sent', '')
        if len(query_sent) == 0:
            self.write('please enter sentence info.')
            return

        if query_sent[0].isalnum() is False:
            self.write('please enter english sentence.')
            return

        sent_list = []
        sent_list.append(query_sent)

        res = bert_client.encode(sent_list)
        res_vec = res[0].tolist()

        top_k = self.get_argument('top', '3')
        ret, result_str = caiss.sync_search(handle, res_vec, CAISS_SEARCH_QUERY, int(top_k), 0)
        if 0 != ret:
            self.write('search failed for the reason of : ' + ret)
            return

        result_dict = json.loads(result_str)
        sent_list = list()
        for info in result_dict['details']:
            sent_list.append(info['label'])

        self.write('the query sentence is [' + query_sent + '].')
        self.write('<br>')
        self.write('the info you also want to know maybe : ')
        self.write('<br>')
        for i in sent_list:
            self.write('****' + i)
            self.write('<br>')


def make_app():
        return tornado.web.Application([
            (r"/", MainHandler),
            (r"/caiss/word", CaissWordHandler),
            (r'/caiss/sentence', CaissSentenceHandler)
        ])


def server_start():
    app = make_app()
    app.listen(8881)
    tornado.ioloop.IOLoop.current().start()


def bert_server_start():
    # 感谢哈工大人工智能团队提供的bert服务
    MODEL_PATH = r'/Users/chunel/Documents/code/python/uncased_L-12_H-768_A-12'
    args = get_args_parser().parse_args(['-num_worker', '2',
                                         '-model_dir', MODEL_PATH,
                                         '-port', '5555',
                                         '-port_out', '5556',
                                         '-max_seq_len', 'NONE',
                                         '-mask_cls_sep',
                                         '-cpu'])
    bert_server = BertServer(args)
    bert_server.start()


if __name__ == "__main__":
    # http://127.0.0.1:8888/caiss/word?query=water
    # http://127.0.0.1:8888/caiss/sentence?sent=i am fine
    bert_server_start()    # 开启bert服务
    print('[caiss] bert server start success...')

    bert_client = BertClient()
    print('[caiss] bert client start success...')

    caiss = PyCaiss(LIB_PATH, MAX_THREAD_SIZE, CAISS_ALGO_HNSW, CAISS_MANAGE_SYNC)
    handle = c_void_p(0)
    caiss.create_handle(handle)
    caiss.init(handle, CAISS_MODE_PROCESS, CAISS_DISTANCE_INNER, DIM, SENT_MODEL_PATH)
    print('[caiss] environment init success...')

    server_start()    # 开启tornado服务，对外提供能力

    caiss.destroy(handle)
