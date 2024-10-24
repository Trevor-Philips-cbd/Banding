rm -rf train/*
python new_strip_xml-v2.py
python xml2txt.py
source make_train.sh
python fix_png.py