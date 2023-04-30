import streamlit as st
import cv2 
import numpy as np 
import pandas as pd 
from io import StringIO
import re

st.set_page_config(
    page_title="Mapper Helper",
    page_icon=":red[\u2605]",
    layout="wide",
    initial_sidebar_state="expanded",
   
)
d1 = {"cl":'str',"colour":"str","Province":"str","terrain":"str",
"rgo1":"str","rgo2":"str","rgo3":"str","buildings":"str"}
l0 = {"Entries": "YES","Colour":"NO","Province":"YES","Terrain":"NO"}

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
                        "pop_type":"pt_","culture":"c_","religion":"r_"}
    available_categories = {"buildings" :[],"units":[], "terain":[],
                        "pop_type":[],"culture":[],"religion":[]}

    for x,y in zip(categories_for_search.keys(),categories_for_search.values()):
        result = re.findall(rf"(?<=\n){y}.*(?==)", needed_content)
        available_categories[x].extend(result)
    return available_categories 


def save_changes(df):
    with open("provinces_new.lua",'w') as file:
        st.write(" len ",len(df["Province"]))
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

        

def main(): 
    if "first_run" not in st.session_state:
        st.session_state.processed1 = False
        st.session_state["first_run"] = True
        st.session_state["colour"] = "#00ff00"

    st.session_state.second_phase_disabled = True
    #df0 = pd.DataFrame([["NO" for x in list(d1.keys())]],columns=list(d1.keys()))
    df0 = pd.DataFrame(l0,index=[0])
    columns = st.columns(3)
    with columns[1]:
        st.title(":red[\u2605] Mapper Helper :red[\u2605]")
        st.write('<p style="text-align: center">v0.1 Workers of the world unite! </p>',
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
            cl_cont1 = st.columns([2,1,2])

            with cl_cont1[0]:
                st.write("<b>New province table </b>",unsafe_allow_html=True) 
                df  = pd.read_csv(file3,dtype=d1) 
                df["terrain"] = (
                df["terrain"].astype("category").cat.add_categories(a_cat["terain"]))

                edited_df = st.experimental_data_editor(df,num_rows="dynamic")
                #st.write(update_verification(df0,edited_df))
                if update_verification(df0,edited_df):
                    st.session_state.second_phase_disabled = False
                st.write("<b>Verification table unlocks save button</b>",
                         unsafe_allow_html=True)
        
                st.dataframe(df0.style.applymap(color_yes_no))
       
            with cl_cont1[2]:
                urgb = unique_colours(file2) # works 
                
                color = st.color_picker(label="**Pick next province colour**")
                st.write(" picked colour is ",color)  
                scl = re.sub("#","",color)
        
                if scl.lower() in urgb or scl.upper() in urgb:
                    st.warning(":red[ COLOUR IS ALREADY IN USE !]")
                else:
                    st.success(":green[ COLOUR IS FREE FOR USE !]")
                st.write(st.session_state)

            with cl_cont1[1]:
                
                df1 = pd.DataFrame( edited_df["colour"])

                st.write("<b> Used colours </b>",unsafe_allow_html=True) 
                st.dataframe(df1.style.applymap(color_as_input))

                if st.button(" SAVE CHANGES ",disabled=st.session_state.second_phase_disabled):
                    save_changes(edited_df)

    st.warning(":red[ Demo part under this line, not included in save at the moment ]")
    # this all needs to be conditional on file loaded and passed checks !

    with st.container():
        cl_cont2 = st.columns(3)
        with cl_cont2[0]:
            rad = st.radio("Additional province input ",options=["NO","YES"],horizontal=True,
            disabled=st.session_state.second_phase_disabled)
        if all((file1,file2,file3)) and rad == "YES":
            with cl_cont2[1]:
                st.selectbox(label="Province to Edit",options=edited_df["Province"])
    if all((file1,file2,file3)) and rad=="YES":
        with st.container():
            cl_cont3 = st.columns(3)
            with cl_cont3[0]:
                df_pop = pd.DataFrame(data=np.zeros((len(a_cat["pop_type"]),2),dtype=np.float64),
                     index=a_cat["pop_type"],columns=["number","literacy"])
                df_pop["number"]=df_pop["number"].astype(np.int64) 
                ede_1 = st.experimental_data_editor(df_pop)
            with cl_cont3[1]:
                df_bul = pd.DataFrame({"buildings":"","buildings_lvl":1},index=[0])
                df_bul["buildings"] =(df_bul["buildings"].astype("category").cat.add_categories(a_cat["buildings"])) 
                ede_2 = st.experimental_data_editor(df_bul,num_rows="dynamic")
        # add cores / called nucleus  / optional
        # assign to country / optional    
if __name__ == "__main__":
    main()
