import os
import pandas as pd
import xml.etree.ElementTree as ET


dir_tifs = "/home/tobias/tmp/derma/wsi/"
dir_results = "/home/tobias/tmp/derma/results/2019-10-17_22-04-35/"
dir_preannos = "/home/tobias/tmp/derma/anno_pre/"
dir_out = "/home/tobias/tmp/derma/wsi/"

dir_tifs = "/home/nikolay/data_ssd/nagelpilz_p150/wsi/"
dir_results = "/home/nikolay/medtorch/resultsVali/2019-10-17_22-04-35/"
dir_preannos = "/home/nikolay/data_ssd/nagelpilz_p150/anno_pre/"
dir_out = "out/"


threshold = 0.6



# --------------------------------------------------------------------------------------------------------------


cases = []

for i in range(100, 200):
    for t in ["N", "P"]:
        case = "{}{}".format(t, i)
        filename_probs = os.path.join(dir_results, "probPatch{}.tif.csv".format(case))
        if os.path.exists(filename_probs):
            cases.append(case)


print("#cases", len(cases))


def _read_xml_points(filename_xml):
    s = ""
    tree = ET.parse(filename_xml)
    for x in tree.iter("Annotation"):
        s += str(ET.tostring(x, encoding="unicode")) + '\n'
    return s


def _get_color(p):
    if p>0.9:
        return "#FF0000" #red
    elif p>0.8:
        return "#FF9900" #orange
    elif p>0.7:
        return "#00FF00" #green
    else:
        return "#1b5e20" #dark green


def write_rectangle_xml(df, save_xml, w=256, xml_preanno=None):
    s = '<?xml version="1.0"?>' + '\n'
    s += '<ASAP_Annotations>' + '\n'
    s += '<Annotations>' + '\n'

    if xml_preanno is not None:
        print("include preanno")
        s += _read_xml_points(xml_preanno)

    for idx in df.index:
        row = df.loc[idx]
        x = row["x0"]
        y = row["y0"]
        p = row["p"]

        x0 = x - w//2
        x1 = x + w // 2
        y0 = y - w // 2
        y1 = y + w // 2

        s += '<Annotation Name="Annotation {}%" Type="Rectangle" PartOfGroup="None" Color="{}">'.format(int(100*p),  _get_color(p)) + '\n'
        s += '<Coordinates>' + '\n'
        s += '<Coordinate Order="0" X="{}" Y="{}" />'.format(x0, y0) + '\n'
        s += '<Coordinate Order="1" X="{}" Y="{}" />'.format(x1, y0) + '\n'
        s += '<Coordinate Order="2" X="{}" Y="{}" />'.format(x1, y1) + '\n'
        s += '<Coordinate Order="3" X="{}" Y="{}" />'.format(x0, y1) + '\n'
        s += '</Coordinates>' + '\n'
        s += '</Annotation>' + '\n'

    s += '</Annotations>' + '\n'
    s += '<AnnotationGroups />' + '\n'
    s += '</ASAP_Annotations>' + '\n'

    f = open(save_xml, "w")
    f.write(s)
    f.close()



for c in cases:
    print("=== {}".format(c))

    filename_probs = os.path.join(dir_results, "probPatch{}.tif.csv".format(c))
    filename_tif = os.path.join(dir_tifs, "{}.tif".format(c))

    df_probs = pd.read_csv(filename_probs, names=["x0", "y0", "p"])
    df_probs["x0"] = df_probs["x0"].astype(int)
    df_probs["y0"] = df_probs["y0"].astype(int)


    # --------------------------------------------------------------------------------------------------

    xml_preanno = os.path.join(dir_preannos, "{}.xml".format(c))
    xml_preanno = xml_preanno if os.path.exists(xml_preanno) else None
    write_rectangle_xml(df_probs[df_probs["p"]>threshold],
                        save_xml=os.path.join(dir_out, "{}.xml".format(c)),
                        xml_preanno=xml_preanno)


