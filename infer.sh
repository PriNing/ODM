python3 tools/visualize.py  \
--model_path /path/to/model \
--char_dict_pth tools/oclip_char_dict  \
--model_config_file src/training/model_configs/RN50_Seg_Clip.json  \
--img_path ./demo/sample.jpg \
--img_text The text to be reconstructed \  #For example:STIRLING CASTLE
--save_path /path/to/save \