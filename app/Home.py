import io
from io import BytesIO
import base64

import pandas as pd
import tabula
import PyPDF2
from img2table.document import Image
import camelot
from importlib import reload
import streamlit as st

import ocrmypdf

# Set page title
st.set_page_config(page_title='IRM App')
st.title("IRM Web App - Home Page")


# @st.cache_data(ttl=60*60)
# def load_data():
#     df = utils.read_from_s3(file_name='scrapings/lcc.csv')
#     df['capacity'] = df['capacity'].astype(int)
#     return df

# # Sidebar filters
# def get_page():
#     page = st.sidebar.number_input(label='PDF Page',min_value=1,max_value=100)
#     return page

# def get_ocr_choice():
#     ocr_choice = st.sidebar.selectbox('Tabula/Camelot',['Tabula','Camelot'])
#     return ocr_choice.lower()


# # Main function
# def pdf_get_df(pdf_file,convert_image_to_pdf=False,full_page=True,lib='camelot',pages='all'):

#     if convert_image_to_pdf:
#         input_file = pdf_file
#         output_file = input_file.replace('.pdf','_clean.pdf')
#         print('running ocrmypdf')
#         print('placeholder here')
#         print('ran ocrmypdf')
#         pdf_file = output_file

#     print(f'pdf file: {pdf_file}')

#     if lib=='tabula':
#         df_list = tabula.read_pdf(pdf_file, pages=pages)
        
#     elif lib=='camelot':
#         if full_page:
#             df_list = [t.df for t  in camelot.read_pdf(pdf_file, pages=pages,flavor='stream',table_regions=['100,350,550,250'],strip_text='\n')]
#         else:
#             df_list = [t.df for t in camelot.read_pdf(pdf_file, pages=pages,flavor='stream',strip_text='\n')]

#     df_list = [d.dropna(how='all') for d in df_list]

#     print(f'number of tables on pages {pages}: {len(df_list)}')

#     return df_list

# def show_pdf(uploaded_file):
#     # with open(file_path,"rb") as f:
#     with io.BytesIO() as buffer:
#         buffer.write(uploaded_file.read())
#         buffer.seek(0)
#         base64_pdf = base64.b64encode(buffer.read()).decode('utf-8')

#         pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
#         st.markdown(pdf_display, unsafe_allow_html=True)

# def main():
#     # Set the title and layout of the app
#     st.title('Financial Statement Parser')

#     # Add content to the home page
#     st.write('Upload a PDF file to parse the financial statement.')

#     # Create a file uploader
#     uploaded_file = st.file_uploader('Upload PDF', type='pdf')

#     # Check if a file has been uploaded
#     if uploaded_file is not None:
#         file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type}

#         # Read the selected page
#         show_pdf(uploaded_file)

#         page = get_page()
#         # Call the function to parse the PDF and get a DataFrame
#         df_list = pdf_get_df(pdf_file=uploaded_file,
#                     convert_image_to_pdf=0,
#                     pages=str(page),
#                     lib=get_ocr_choice(),
#                     full_page=0)
        
#         for c,df in enumerate(df_list):
#             # print(c)
#             st.dataframe(df_list[c])
 
# # Run the app
# if __name__ == '__main__':
#     main()

