#coding:utf-8
f = open('log.txt','rb')
res = f.readlines()
f.close()
f = open('log.txt','w')
for pt in res:
    pt = str(pt, encoding='utf-8')
    if pt.find('error')>=0 and pt.find('_error')<0 and pt.find('error_')<0:
        f.write(pt)
f.close()