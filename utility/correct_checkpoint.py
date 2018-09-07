'''
Created on Jun 29, 2018

@author: en
'''
import os, sys

def should_correct_cpt(path):
    
    txt = open('%s/checkpoint'%path).readlines()[0]

    tmp = txt.split('"')
    
    if os.path.exists(tmp[1]+'.index'):

        if path == tmp[1].split('/model.ckpt')[0]:
            return False
        else:
            return True
    return True

def correct_path(path):
    path = os.path.abspath(path)
    txt = open('%s/checkpoint'%path).readlines()
    with open('%s/checkpoint'%path, 'w') as f:
        for each in txt:
            first, cpt_path, end = each.split('"')
            fld_path, _ = cpt_path.split('/model.ckpt')
            cpt_path = cpt_path.replace(fld_path, path)
            f.write('%s"%s"%s'%(first, cpt_path, end))

if __name__ == '__main__':
#    fd = '/data/chercheurs/en/campose/networks/ShopFacade/RelativePoseNetRT/gridsearch/'
    fd = sys.argv[1] #'/data/chercheurs/en/campose/networks/ShopFacade/RelativePoseNetRT/gridsearch/'
    
    for each_folder in os.listdir(fd):
        if len(each_folder) < 3: continue
        tmp_path = fd + '/'+ each_folder
        for each_train in os.listdir(tmp_path):
            if should_correct_cpt('%s/%s/'%(tmp_path, each_train)):
                print('corrected checkpoint path at', each_folder, each_train)
                correct_path('%s/%s/'%(tmp_path, each_train))
            
    
    
