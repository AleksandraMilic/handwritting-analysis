import xml.etree.ElementTree as ET
import pandas as pd 
import glob

lineStroke = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes\\*'

def parse_xml(path_xml):
    '''child: strokeSet
       subchild: Stroke and time..'''
    # x_coord = []
    # y_coord = []
    # time = []

    # tree = parse(path_xml)
    # root = tree.getroot()
    # childs_list = []
    
    # for child in root:     
    #     childs_list.append(child.tag) #child.tag

    # strokeSet = childs_list[1] 
    df_covid = pd.DataFrame()
    xml_data = open(path_xml, 'r').read()  # Read file
    xml_name = path_xml[86:]
    xml_name =  xml_name[:-4:]
    print(xml_name)

    df = pd.read_xml(xml_data, xpath=".//Point")
    print(df)
    df.head()
    df_covid = pd.concat([df_covid, df])
    df_covid.to_csv('D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\'+ xml_name +'.csv')
    
    

    

    
    return




if __name__ == "__main__":
        
    for a01 in  glob.glob(lineStroke):
        for a01_000 in glob.glob(a01+"\\*"):
            for path_xml in glob.glob(a01_000+"\\*.xml"):
                parse_xml(path_xml)
    # path_xml='D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes\\a01\\a01-000\\a01-000u-01.xml'
    # parse_xml(path_xml)
                

