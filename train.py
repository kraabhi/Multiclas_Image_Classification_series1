# The very first step is to get Data and place in the required location to be used while using fastai.
# !wget https://storageCharacters_Train.zip (hidden link)
# unzip the data and place in data tab:  ! unzip -qq Characters_Train.zip -d data/
# Defined the path variable to be used in fast.ai by using untar_data
path = untar_data("/content/data/Characters_Train/")

# create csv containing image name and category

sub_dir= os.listdir('/content/data/Characters_Train')
input_dir         = "/content/data/Characters_Train"
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
        final_list.append({'Image': os.path.join("Characters_Train/",sub,img),'category':sub})

output_df = pd.DataFrame(final_list)
output_df.to_csv(output_file, index=False)

# check the csv created
df = pd.read_csv(path/'labels.csv')

# resnet 50 training
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_csv(path, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)
                                  
# its best to see data
data.show_batch(rows=3, figsize=(5,5))
# check the number of classes to verify
data.c
learn = cnn_learner(data, models.resnet50,metrics = error_rate )
learn.fit_one_cycle(1) # just training for 1 epoch por practice to give an idea
learn.save("model1)  #save first model
