import json

import tornado.ioloop
import tornado.web

from python.pyCaiss import *

LIB_PATH = r'/Users/chunel/Documents/code/cpp/caiss/python/libCaiss.dylib'
MODEL_PATH = r'/Users/chunel/Documents/code/cpp/models/bert_71290words_768dim.caiss'
MAX_THREAD_SIZE = 1
DIM = 768
TOP_K = 5


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, welcome to the world of Caiss")

class CaissHandler(tornado.web.RequestHandler):
    def get(self):
        query_word = self.get_argument('query', '')
        if len(query_word) == 0:
            self.write('please enter query word')
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


def make_app():
        return tornado.web.Application([
            (r"/", MainHandler),
            (r"/caiss", CaissHandler),
        ])

def server_start():
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    # http://127.0.0.1:8888/caiss?query=water
    caiss = PyCaiss(LIB_PATH, MAX_THREAD_SIZE, CAISS_ALGO_HNSW, CAISS_MANAGE_SYNC)

    handle = c_void_p(0)
    caiss.create_handle(handle)
    caiss.init(handle, CAISS_MODE_PROCESS, CAISS_DISTANCE_INNER, DIM, MODEL_PATH)

    server_start()

    caiss.destroy(handle)
