import wget
 
url = "https://storage.googleapis.com/kaggle-data-sets/1500837/2491748/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240630%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240630T085648Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=134f7282dc75a3e7e13866d878e7b4c154e61d0adc4047fdffd226d479c4501417649faa3b8b69489c0257e93f1cee4185ca431598d50d6fbdea84c5bd7867acd75ef1a75c91e0bf03b7e5fcc7aac49539f440a3e0bb61e9c1536464b1b0f8e5dcfc8c1ff278a5a8eb8c90a74ddc1c5756520dd9208eceaa7350ad43dbc501f889fdec9e60b1fbc04f583ced621b06daa7e8852bf51e54e4f78a176a7845e278ebb15fdbd2d7bb3415acfec153da0410378ee38bfe33665ede5475800729a2f4cf4918a44eff1b9fe576f2c462e46d24bff7d2f396a22e32d933aa568edca49ed4148e096e323f6dfe176e9b1b694e194349fa40971ae979fcb0bf0582970780"
output_filename = 'IN100.zip'  # 文件名
 
# 下载到和.py同路径
wget.download(url, out=output_filename)
 
# 下载到其他路径
# out=路径+文件名
# out='./tmp/'+output_filename