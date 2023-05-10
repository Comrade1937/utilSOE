import streamlit as st
import cv2 
import numpy as np 
import pandas as pd 
from io import StringIO
import re
import json 
from streamlit.elements.data_editor import _apply_dataframe_edits
from streamlit import type_util

st.set_page_config(
    page_title="Mapper Helper",
    page_icon=":red[\u2605]",
    layout="wide",
    initial_sidebar_state="expanded",
   
)
d1 = {"cl":'str',"colour":"str","Province":"str","terrain":"str",
"rgo1":"str","rgo2":"str","rgo3":"str","buildings":"str"}
l0 = {"Entries": "YES","Colour":"NO","Province":"YES","Terrain":"NO"}
def ff(name,dfindex,key):
    t = st.session_state.dfl[name][dfindex]
    t = type_util.convert_anything_to_df(t)
    _apply_dataframe_edits(t, st.session_state[key])
    st.session_state.dfl[name][dfindex] = t 


def color_yes_no(v):
    if v == "YES":
        return "background-color: green"
    else : 
        return "background-color: red"
def color_as_input(v):
    if v is np.nan  or v is None:
        return "background-color: white"
    if  re.match(r"#.*",v):
        return f"background-color : {v}"
    else :
        return "background-color: white" 

def is_list_of_hex_colours(l) -> bool:
     return all([ re.fullmatch(r'#[0-9a-fA-F]{6}',x) for x in l])

def update_verification(vdf,df) -> bool:
    if len(df["Province"])==0:
        vdf["Province"]= "NO"
        " At least terain and colour must be filled to have minimal Province entry!"
    else :
        vdf["Province"] = "YES"
    if len(df["colour"]) == 0 :
        vdf["Entries"] = "NO"
        st.write(" You need at least one valid entry !")
    else: 
        vdf["Entries"] = "YES"
        if df["colour"].isna().any():
            vdf["Colour"][0] = "NO"
            st.write(" You need to have all colours filled!")
        else:
            if is_list_of_hex_colours(list(df["colour"])):
                if len(df["colour"])== len(set(df["colour"])):
                    vdf["Colour"][0] = "YES"
                else:
                    vdf["Colour"][0] = "NO"
                    st.write("One or more colours are repeating!") 
            else: 
                vdf["Colour"][0] = "NO"
                st.write(" Some of colour fields are not valid hex colours!")
    if len(df["terrain"]) == 0 :
        vdf["Terrain"] = "NO"
        st.write(" Terrain field cannot be empty")
    else:
        if df["terrain"].isna().any():
            vdf["Terrain"] = "NO"
            st.write(" Fill EACH terain field")
        else: 
            vdf["Terrain"] = "YES"
    
    return all([x == "YES" for x in vdf.iloc[0]])

@st.cache_data
def unique_colours(uploaded_file):

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img1 = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

    vertical_img = img.reshape((-1,3))
    vertical_24 = np.dot(vertical_img.astype(np.uint32),[1,256,65536])
    vunique24 = np.unique(vertical_24)

    C1 = np.bitwise_and(vunique24, 0xff)      
    C2 = np.bitwise_and(np.right_shift(vunique24,8),  0xff)   
    C3 = np.bitwise_and(np.right_shift(vunique24,16), 0xff)   
    unique_rgb = np.stack((C1, C2, C3),axis=-1)

    rgb2hex = lambda r,g,b: '%02x%02x%02x' %(r,g,b)
    urgb = [ rgb2hex(*i) for i in unique_rgb]
    return urgb 


@st.cache_data
def read_lua(uploaded_file):
    sfile = StringIO(uploaded_file.getvalue().decode("utf-8"))
    read_content = sfile.read()
  
    result = re.search(r"province=",read_content)
    last_declaration_position = result.span()[0]
    needed_content = read_content[:last_declaration_position]
    categories_for_search = { "buildings" :"bt_","units":"ut_", "terain":"tt_",
                        "pop_type":"pt_","culture":"c_","religion":"r_","nation":"n_"}
    available_categories = {"buildings" :[],"units":[], "terain":[],
                        "pop_type":[],"culture":[],"religion":[],"nation":[]}

    for x,y in zip(categories_for_search.keys(),categories_for_search.values()):
        result = re.findall(rf"(?<=\n){y}.*(?==)", needed_content)
        available_categories[x].extend(result)
    return available_categories 

def write_province_table(fl,name,terrain,color,dfl2) ->None: 
    s1 = f'province=Province:new{{ref_name="{name}",'
    scl = re.sub("#","0x",color)
    s2 = f'name=translate("{name.capitalize()}"),color={scl},'
    s3 = f'terrain={terrain},'
    s4 = write_rgo_table(dfl2)
    s5 = s1+s2+s3+s4+"}\n"
    fl.write(s5)
    fl.write("province:register()\n") 

# 1. write pop table - no checks just default 0,0.0 
def write_pop_table(fl,df) -> None:
    s1 ="province:add_pop("
    for i in range(len(df["number"])):
        if df.iloc[i,1]!= 0 : # drop 0 to save file size
            s2 = s1+f"{df.iloc[i,0]}"+f",{df.iloc[i,1]},{df.iloc[i,2]})\n"
            fl.write(s2)

# 2. write buildings table  - needs to siletnly drop buildings that are none 
def write_buildings_table(fl,df)->None:
    s1 = "province:create_building("
    for i in range(len(df["buildings"])):
        if df.iloc[i,0] not in [None,np.nan]:
            s2 = s1 + f"{df.iloc[i,0]},{df.iloc[i,1]})\n"
            fl.write(s2)
# 3. create rgo string - since rgo is part of header - silent drop of none 
def write_rgo_table(df) ->str: 
    if df["rgo"].count() == 0 :
        return "rgo_size={{},}"
    else:
        s1 ="rgo_size={" 
        s3 = "}"
        s2 =""
        for i in range(len(df["rgo"])):
            if df.iloc[i,0] not in [None,np.nan]:
                s2 = s2 + f'{{"{df.iloc[i,0]}",{df.iloc[i,1]}}},'
        return s1+s2+s3 

# 4. write cores table  - again silent drop of none 
def write_core_table(fl,df)->None:
    s1 = "province:add_nucleus("
    for i in range(len(df["cores"])):
        if df.iloc[i,0] not in [None,np.nan] :
            s2 = s1+f"{df.iloc[i,0]})\n"
            fl.write(s2)

# 5. write owner table - DANGER - WARNING - no check !!!
def write_owner_and_capital(fl,df)->None:
    s1 ="province:give_to("
    own = df["owner"][0]
    if own not in [None, np.nan]:
        s2 = s1 + str(own)+")\n" 
        fl.write(s2)
        if df["capital"][0] == True :
            s1a = ":set_capital(province)\n"
            s2a = str(own)+s1a
            fl.write(s2a)
  
# 6. write language table - silent drop of nan 
def write_language_table(fl,df)->None :
    s1= "province:set_language("
    for i in range(len(df["language"])):
        if df.iloc[i,0] not in [None,np.nan]:
            s2 = s1 + f"{df.iloc[i,0]},{df.iloc[i,1]})\n"
            fl.write(s2)

# 7. write religion table - silet drop if 0.
def write_religion_table(fl,df)->None:
    s1 = "province:set_religion("
    for i in range(len(df["religion"])):
        if df.iloc[i,1]>0. :
            s2 = s1 + f"{df.iloc[i,0]},{df.iloc[i,1]})\n"
            fl.write(s2)

def save_changes(df,gui_element):
    if st.session_state["rad"] == "NO":
        with open("provinces_new.lua",'w') as file:
            for i in range(len(df["Province"])):
                # TO DO check for empty string , na , None     
                sa = df["Province"].values[i]
                sb = df["colour"].values[i]
                sb = re.sub("#","0x",sb)
                sc = df["terrain"].values[i]
                cc=f'province=Province:new{{ref_name="{sa}",name=translate("{sa.capitalize()}"),color={sb.lower()},terrain={sc},rgo_size={{{{}},}}}}'
                file.write(cc)
                file.write("\n")
                file.write("province:register()")
                file.write("\n")
            with gui_element:
                st.success("FILE SAVED")
    else: # detail option
        bt = []
        for x in df["Province"] : # so for entry in list of added provinces 
            bt.append(pop_table_check(st.session_state.dfl[x][0], x, gui_element))
            bt.append(buildings_table_check(st.session_state.dfl[x][1], x, gui_element))
            bt.append(rgo_table_check(st.session_state.dfl[x][2], x, gui_element))
            bt.append(core_table_check(st.session_state.dfl[x][3], x, gui_element))
            bt.append(language_table_check(st.session_state.dfl[x][5], x, gui_element))
            bt.append(religion_table_check(st.session_state.dfl[x][6], x, gui_element))
        if not all(bt):
            with gui_element:
                st.error("FILE NOT WRITTEN !")
        else:
            with open("provinces_new.lua",'w') as file:
                for i,x in enumerate(df["Province"]) : 
                    write_province_table(file, x,df.iloc[i,1],df.iloc[i,2], st.session_state.dfl[x][2])
                    write_buildings_table(file,st.session_state.dfl[x][1])
                    write_pop_table(file,st.session_state.dfl[x][0])       
                    write_language_table(file,st.session_state.dfl[x][5] )
                    write_religion_table(file,st.session_state.dfl[x][6])
                    write_core_table(file,st.session_state.dfl[x][3])
                    write_owner_and_capital(file,st.session_state.dfl[x][4])
                with gui_element:
                    st.success("FILE WRITTEN")
# CHECK FUNCTIONS FOR TABLES 

# 1. pop table  , N >0 , 0. <= fr <=1.
def pop_table_check(df,province,gui_element) -> bool:
    pass_num = all(np.where(df['number'] >= 0, True, False))
    pass_fr =  all(df['literacy'].apply(lambda x: True if ((x>=0.) & (x<=1.)) else False))
    if pass_num and pass_fr :
        return True 
    else: 
        with gui_element:
            st.write(f"not allowed entry in {province} pop table, pls FIX!")
        return False 
    
# 2. building table   1<= N <= building_lvl from json but only for rows where building is selected 
def buildings_table_check(df,province,gui_element) -> bool:
    pass_build = True 
    for i in range(len(df["buildings"])):
        if df["buildings"].iloc[i] not in [None,np.nan]:
            if not(df["lvl"].iloc[i]>=1 and df["lvl"].iloc[i]<=st.session_state.dlimit["building_lvl"]):
                pass_build = False
    if not pass_build :
        with gui_element:
            st.write(f"Not allowed entry in {province} buildings table, pls FIX!")
    return pass_build
# 3. rgo table  again for non empty 1<= N <= rgo_lvl from json , for non empty rows
def rgo_table_check(df,province,gui_element) -> bool:
    pass_rgo = True 
    for i in range(len(df["rgo"])):
        if df["rgo"].iloc[i] not in [None,np.nan]:
            if not(df["lvl"].iloc[i]>=1 and df["lvl"].iloc[i]<=st.session_state.dlimit["rgo_lvl"]):
                pass_rgo = False
    if not pass_rgo :
        with gui_element:
            st.write(f"Not allowed entry in {province} rgo table, pls FIX!")
    return pass_rgo
# 4 cores table check - nothing to check at the moment  maybe double entries ?
def core_table_check(df,province,gui_element) -> bool:
    df1 = df.dropna()
    pass_core = True 
    if len(df1["cores"]) != len(df1["cores"].unique()) :
        pass_core = False
        with gui_element:
            st.write(f"Repeating entries in {province} core table, pls FIX!")
    return pass_core
# 5.owner nothint to check  for now - NEEDS check for double assignment over all the lua hmm
# WARNING

# 6. language table check - double entries and fraction for entry 
def language_table_check(df,province,gui_element) -> bool:
    df1 = df.dropna()
    pass_double_lang = True 
    if len(df1["language"]) != len(df1["language"].unique()) :
        pass_double_lang = False
        with gui_element:
            st.write(f"Repeating entries in {province} language table, pls FIX!")
    pass_frac = True 
    for i in range(len(df["fraction"])):
        if df["fraction"].iloc[i] not in [None,np.nan]:
            if not(df["fraction"].iloc[i]>=0.0 and df["fraction"].iloc[i]<=1.0):
                pass_frac = False
    if not pass_frac:
        with gui_element:
            st.write(f"Not allowed entries in {province} language table, pls FIX!")
    return pass_double_lang and pass_frac
# 7. religion table check - double entries and fraction for entry 
def religion_table_check(df,province,gui_element) -> bool:
    df1 = df.dropna()
    pass_double_rel = True 
    if len(df1["religion"]) != len(df1["religion"].unique()) :
        pass_double_rel = False
        with gui_element:
            st.write(f"Repeating entries in {province} religion table, pls FIX!")
    pass_frac = True 
    for i in range(len(df["fraction"])):
        if df["fraction"].iloc[i] not in [None,np.nan]:
            if not(df["fraction"].iloc[i]>=0.0 and df["fraction"].iloc[i]<=1.0):
                pass_frac = False
    if not pass_frac:
        with gui_element:
            st.write(f"Not allowed entries in {province} religion table, pls FIX!")
    return pass_double_rel and pass_frac



def main(): 
    if "limits" not in st.session_state:
        st.session_state.limits = True
        with open("limits_mapperhelper.json","r") as file:
            data = json.load(file)
            st.session_state.dlimit = data 

    st.session_state.second_phase_disabled = True

    df0 = pd.DataFrame(l0,index=[0])
    columns = st.columns(3)
    with columns[1]:
        st.title(":red[\u2605] Mapper Helper :red[\u2605]")
        st.write('<p style="text-align: center">v0.2.1 Workers of the World unite! </p>',
                unsafe_allow_html=True)
   
    with st.sidebar:
        st.write(" START HERE ")
        file1 = st.file_uploader("**Upload province.lua**",
                                accept_multiple_files=False,type="lua")
        file2 = st.file_uploader("**Upload province.png**",
                                accept_multiple_files=False,type="png")
        file3 = st.file_uploader("**Upload your.csv**",
                                accept_multiple_files=False,type="csv")
    if not all((file1,file2,file3)):
       
        st.warning(" :red[Not ready, pritty please upload all the files !] ")
    if all((file1,file2,file3)):
        a_cat = read_lua(file1)
        st.success(" :green[Tnx for loading files, lets Map] ")

        with st.container():
            cl_cont1 = st.columns([2,6,6,5])

            with cl_cont1[1]:
                st.write("<b>New province table </b>",unsafe_allow_html=True) 
                df  = pd.read_csv(file3,dtype=d1) 
                df["terrain"] = (
                df["terrain"].astype("category").cat.add_categories(a_cat["terain"]))

                edited_df = st.experimental_data_editor(df,num_rows="dynamic")
                
                if update_verification(df0,edited_df):
                    st.session_state.second_phase_disabled = False
                st.write("<b>Verification table unlocks save button</b>",
                         unsafe_allow_html=True)
        
                st.dataframe(df0.style.applymap(color_yes_no))
       
            with cl_cont1[3]:
                urgb = unique_colours(file2)  
                color = st.color_picker(label="**Pick next province colour**")
                st.write(" picked colour is ",color)  
                scl = re.sub("#","",color)
        
                if scl.lower() in urgb or scl.upper() in urgb:
                    st.warning(":red[ COLOUR IS ALREADY IN USE !]")
                else:
                    st.success(":green[ COLOUR IS FREE FOR USE !]")
                #st.write(st.session_state)

            with cl_cont1[2]:
                
                df1 = pd.DataFrame( edited_df["colour"])

                st.write("<b> Used colours </b>",unsafe_allow_html=True) 
                st.dataframe(df1.style.applymap(color_as_input))
                btn = st.button(" SAVE CHANGES ",disabled=st.session_state.second_phase_disabled)
                conti = st.container()
                if btn:
                    save_changes(edited_df,conti)
        

    with st.container():
        cl_cont2 = st.columns(3)
        with cl_cont2[0]:
            rad = st.radio("Additional province input ",options=["NO","YES"],horizontal=True,
            disabled=st.session_state.second_phase_disabled,key="rad")
        if all((file1,file2,file3)) and rad == "YES":
            with cl_cont2[1]:
                choosen_province = st.selectbox(label="Province to Edit",options=edited_df["Province"])
    if all((file1,file2,file3)) and rad=="YES":
        if ("detail_entry" not in st.session_state):
            st.session_state["detail_entry"] = True 
            dfl ={}
            for  i,x in enumerate(edited_df["Province"]):
                # 1. pops table 
                lpops = a_cat["pop_type"]
                lnum  = [0 for i in lpops]
                llit = [0.0 for i in lpops]      
                f1 = pd.DataFrame({"pops":lpops,"number":lnum,"literacy":llit}) 
                f1["pops"] = f1["pops"].astype("category")
                #2. buildings table  
                lbuild = a_cat["buildings"]
                lbuild_blank = [None for i in range(st.session_state.dlimit["building_num"])]
                llvl = [0 for i in range(st.session_state.dlimit["building_num"])]
                f2 = pd.DataFrame({"buildings":lbuild_blank,"lvl":llvl})  
                f2["buildings"]= f2["buildings"].astype("category").cat.add_categories(lbuild)
                #3. rgo table 
                lrgo = [None for i in range(st.session_state.dlimit["rgo_num"])]
                lrgo_lvl = [0 for i in range(st.session_state.dlimit["rgo_num"])]
                f3 = pd.DataFrame({"rgo":lrgo,"lvl":lrgo_lvl}) 
                lrgo_cat = st.session_state.dlimit["rgo"]
                f3["rgo"] = f3["rgo"].astype("category").cat.add_categories(lrgo_cat)
                #4. cores table 
                lcore_lua = sorted(a_cat["nation"])
                lcores = [None for i in range(st.session_state.dlimit["core_num"])]
                f4 = pd.DataFrame({"cores":lcores},dtype="category") # cores   
                f4["cores"] = f4["cores"].astype("category").cat.add_categories(lcore_lua)
                #5. owner table 
                f5 = pd.DataFrame({"owner":[None],"capital":[False]}) # 
                f5["owner"] = f5["owner"].astype("category").cat.add_categories(lcore_lua)
                #6. language table  "unlimited" but limited in json  
                llang = [None for i in range(st.session_state.dlimit["lang_num"])]
                llang_fr = [0.0 for i in range(st.session_state.dlimit["lang_num"])]
                f6 = pd.DataFrame({"language":llang,"fraction":llang_fr})
                llang_cat = sorted(a_cat["culture"])
                f6["language"] = f6["language"].astype("category").cat.add_categories(llang_cat)
                # religion_table  limited by number of religions
                lrel = sorted(a_cat["religion"])
                lrelf = [0.0 for i in lrel]
                f7 = pd.DataFrame({"religion":lrel,"fraction":lrelf}) 
                f7["religion"] = f7["religion"].astype("category")

                dfl[x] = [f1,f2,f3,f4,f5,f6,f7] # ok list of data frames 
            st.session_state["dfl"] = dfl


        with st.container():
            ccol = st.columns([1,14,12,12,12])
            with ccol[1]:
                st.warning(" literacy [0.0,1.0]")
                st.experimental_data_editor(st.session_state.dfl[choosen_province][0],
                on_change=ff,args=(choosen_province,0,"d0"),key="d0")
                st.warning(f" building lvl [1,{st.session_state.dlimit['building_lvl']}]")
                st.experimental_data_editor(st.session_state.dfl[choosen_province][1],
                on_change=ff,args=(choosen_province,1,"d1"),key="d1")
            with ccol[2]:
                st.warning(f" rgo lvl [1,{st.session_state.dlimit['rgo_lvl']}]")
                st.experimental_data_editor(st.session_state.dfl[choosen_province][2],
                on_change=ff,args=(choosen_province,2,"d2"),key="d2")
                st.experimental_data_editor(st.session_state.dfl[choosen_province][3],
                on_change=ff,args=(choosen_province,3,"d3"),key="d3")
            with ccol[3]:
                st.experimental_data_editor(st.session_state.dfl[choosen_province][4],
                on_change=ff,args=(choosen_province,4,"d4"),key="d4")
                st.warning("fraction [0.0,1.0]")
                st.experimental_data_editor(st.session_state.dfl[choosen_province][5],
                on_change=ff,args=(choosen_province,5,"d5"),key="d5")
            with ccol[4]:
                st.warning("fraction [0.0,1.0]")
                st.experimental_data_editor(st.session_state.dfl[choosen_province][6],
                on_change=ff,args=(choosen_province,6,"d6"),key="d6")





if __name__ == "__main__":
    main()
