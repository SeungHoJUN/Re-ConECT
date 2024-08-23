# Re-ConECT
Re-ConECT (text-based Rehabiliation AI for Continuous functional Evaluation &amp; Customized Training)

![image](https://github.com/user-attachments/assets/a3a7603e-6bd5-4af5-b2e8-c8b9ecf2588d)

The Re-ConECT service can be used by both people who have received a diagnosis and those who have not.

The process is as follows:

### 1. Check if the person has received a diagnosis
### 2. For patients who have received a diagnosis: 
1. The patient's ID is entered.
2. If there are previously saved records, they are retrieved. If not, new data is created.
3. A current functional assessment is conducted, and a comparison with records from the past 7 days is performed.
   - for functional assessment : please refer 4_CUE_T_Manual(in dataset)
5. Decreased functions are identified by comparing with previous records, and appropriate exercises are recommended based on these findings.
6. The frequency of exercises and weekly goals are presented.
7. Current symptoms are analyzed comprehensively to determine the possibility of complications, and the need for a hospital visit is indicated.

### 3. For patients who have not received a diagnosis: 
1. An interview is conducted, similar to what is done in an actual rehabilitation hospital.
2. A physical examination is conducted.
3. When the information from steps 1 and 2 is input into the model, it provides a list of expected diagnoses matching the patient's symptoms and necessary tests. Finally, it determines whether the patient needs to go to the hospital.

The code can be easily run in [Re_ConECT.ipynb](https://github.com/SeungHoJUN/Re-ConECT/blob/main/Re_ConECT.ipynb).


For the dataset needed for RAG, please refer to this link: https://drive.google.com/drive/u/0/mobile/folders/1OYvuD7Sq7LDcBhOUBxeaqqzpkZ75oSbj?utm_source=en&pli=1&sort=13&direction=a
