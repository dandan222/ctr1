from googletrans import Translator

translator = Translator(service_urls=['translate.google.ru'])
trans = translator.translate('评定现怀数扑乡心关村村民王培民一家六口人拥有两座小院', src='zh-cn', dest='ru')
# 原文
print(trans.origin)
# 译文
print(trans.text)
