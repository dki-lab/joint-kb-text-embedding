# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# import numpy

# ext_modules = [
#     Extension("data_producer", ["data_producer.pyx"], include_dirs = ['.', numpy.get_include()]),
#     Extension("dictionary", ["wikipedia2vec/wikipedia2vec/dictionary.pyx"], include_dirs = ['.']),
#     Extension("dump_db", ["wikipedia2vec/wikipedia2vec/dump_db.pyx"], include_dirs = ['.']),
#     Extension("mention_db", ["wikipedia2vec/wikipedia2vec/mention_db.pyx"], include_dirs = ['.'])
#     ]

# setup(
#   name = 'app',
#   cmdclass = {'build_ext': build_ext},
#   ext_modules = ext_modules  
# )

#Use this setup.py if you want setup to automatically cythonize all pyx in the codeRootFolder
#To run this setup do exefile('pathToThisSetup.py')

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(
    ext_modules=cythonize(["wiki2vec_cython/wiki2vec/dictionary.pyx",
    						"wiki2vec_cython/wiki2vec/dump_db.pyx",
    						"wiki2vec_cython/wiki2vec/mention_db.pyx",
    						"wiki2vec_cython/wiki2vec/link_graph.pyx",
    						"wiki2vec_cython/wiki2vec/wikipedia2vec.pyx",
    						"wiki2vec_cython/wiki2vec/utils/wiki_page.pyx",
    						"wiki2vec_cython/wiki2vec/utils/wiki_dump_reader.pyx",
    						"wiki2vec_cython/wiki2vec/utils/tokenizer/base_tokenizer.pyx",
    						"wiki2vec_cython/wiki2vec/utils/tokenizer/icu_tokenizer.pyx",
    						"wiki2vec_cython/wiki2vec/utils/tokenizer/jieba_tokenizer.pyx",
    						"wiki2vec_cython/wiki2vec/utils/tokenizer/mecab_tokenizer.pyx",
    						"wiki2vec_cython/wiki2vec/utils/tokenizer/regexp_tokenizer.pyx",
    						"wiki2vec_cython/wiki2vec/utils/sentence_detector/sentence.pyx",
    						"wiki2vec_cython/wiki2vec/utils/sentence_detector/icu_sentence_detector.pyx",
    						"wiki2vec_cython/wiki2vec/utils/tokenizer/token.pyx"]),
    include_dirs=['.', numpy.get_include(),
    'wiki2vec_cython/wiki2vec/']
)