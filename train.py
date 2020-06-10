# !wget https://storageCharacters_Train.zip (hidden link)
! unzip -qq CAX_Characters_Train.zip -d data/
path = untar_data("/content/data/CAX_Characters_Train/")
sub_dir= os.listdir('/content/data/CAX_Characters_Train')
input_dir         = "/content/data/CAX_Characters_Train"
df = pd.DataFrame(list())
df.to_csv('/content/data/labels.csv')
output_file       = "/content/data/labels.csv"
sub_dir           = os.listdir(input_dir)
final_list        = []

for sub in sub_dir:
    #print(sub)
    sub_dir_path  = input_dir+"/"+sub
    images  = os.listdir(sub_dir_path)
    #print(images)
    for img in images:
        #print(img)
        final_list.append({'Image': os.path.join("CAX_Characters_Train/",sub,img),'category':sub})

output_df = pd.DataFrame(final_list)
output_df.to_csv(output_file, index=False)

df = pd.read_csv(path/'labels.csv')

tfms = get_transforms(do_flip=False)

#data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)
data = ImageDataBunch.from_csv(path, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)
                                  
data.show_batch(rows=3, figsize=(5,5))

data.c

learn = cnn_learner(data, models.resnet50,metrics = error_rate )

learn.fit_one_cycle(1)

learn.save("model1)
