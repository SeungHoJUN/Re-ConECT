import pandas as pd
from datetime import datetime, timedelta
from langchain_core.output_parsers import StrOutputParser

def calculate_7day_average(file_path, date=None):
    """
    Calculates the average of data within 7 days from the specified date in the given CSV file.

    :param file_path: Path to the CSV file
    :param date: Reference date (default: None, uses today's date if None)
    :return: List of 7-day average values for each item
    """
    df = pd.read_csv(file_path, parse_dates=['datetime'])

    if date is None:
        date = datetime.now()
    elif isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    seven_days_ago = date - timedelta(days=7)

    df_recent = df[df['datetime'] > seven_days_ago]

    averages = []
    for item in range(1, 18):
        column_name = f'Item {item}'
        avg = df_recent[column_name].mean()
        averages.append(round(avg, 2))

    return averages

def input_item_scores():
    """
    Function to input scores for Items 1 to 17 from the user.
    
    :return: List containing 17 item scores
    """
    scores = []
    for i in range(1, 18):
        while True:
            try:
                score = float(input(f"Enter the score for Item {i}: "))
                scores.append(score)
                break
            except ValueError:
                print("Please enter a valid number.")
    
    return scores

def compare_scores(item_scores, average_scores):
    """
    Compare current scores with 7-day averages and identify items with decreased scores.

    :param item_scores: List of current scores for items 1-17
    :param average_scores: List of 7-day average scores for items 1-17
    :return: List of tuples containing (item name, current score, average score) for decreased items
    """
    item_names = [
        "Reach fwd", "Reach Up", "Reach Down", "Lift Up", "Push Down",
        "Wrist Up", "Acquire - Release", "Grasp Dynamometer", "Lateral Pinch",
        "Pull Weight", "Push Weight", "Container", "Pinch Die", "Pencil",
        "Manipulate (chip)", "Push Index", "Push Thumb"
    ]

    decreased_items = []

    for i, (current, average) in enumerate(zip(item_scores, average_scores)):
        if current < average:
            decreased_items.append(item_names[i])

    return decreased_items

# Function to get patient information from user input
def get_patient_info():
    # Prompt user for each piece of information
    diagnosed_patient = input('chief_complaint: ')
    patient_disability = input("Patient's disability (e.g., Activities using arm): ")
    functional_evaluation = input("Functional evaluation tool (e.g., CUE-T): ")
    new_symptoms = input("Newly acquired symptoms: ")
    
    # Return the information as a list
    return [
        patient_disability,
        functional_evaluation,
        new_symptoms
    ]

def rehabilitation_evaluation(llm, docs, decreased_items, patient_info):
    prompt_template = PromptTemplate.from_template(
        """
        You are a renowned rehabilitation medicine specialist. Evaluate physical functions related to patient's diagnosis and disabilities. Educate the patient with proper rehabilitation exercise with regards to functions declining over time. Check whether there are recently acquired symptoms and check whether those symptoms indicate complications related to patient's diagnosis.

        Functional evaluation and exercise education consists of 3 steps. Answer should be provided in 3 steps.

        - $1 Based on {patient's disability}, suggest which {functional evaluation} is needed for the patient.

        - $2 Suggest body parts used during each item in {ITEMs} with <INTENT> section in {CUE T Manual}.

        - $3 Suggest exercises in {PTX} where "anatomical structures used during each item in {ITEMs}.


        - $4 Get {diagnosed patient} as diagnosis. Check if diagnosis is "Stroke" or "Spinal Cord Injury".
             Get {newly acquired symptoms} as "symptoms".
             - If diagnosis is "Stroke", Show list of extracted complications to patient and recommend hospital visit if "symptoms" indicate certain complications among complications extracted from {Stroke Complications}. Give evidence why "symptoms" indicate extracted complication.
             - Else if diagnosis is "Spinal Cord Injury", Show list of extracted complications to patient and recommend hospital visit if "symptoms" indicate certain complications among complications extracted from {SCI Complications}. Give evidence why "symptoms" indicate extracted complication.

        ---
        Here are the examples of questions and answers between physician and patients.
        ---
        >>Example1<< :

        $Input :
            "diagnosed patient": 'Stroke',
            "patient's disability": 'Activities using arm',
            "functional evaluation": 'CUE-T',
            "ITEMs": ['REACH FORWARD'],
            "newly acquired symptoms":'Pain when moving my weak side shoulders'

        $output :
            required test : CUE T Test.
            Extract anatomical structures : shoulder.

            Item of which the score dropped : 'Reach Forward'
            Exercise recommendation :
            1. Position yourself standing against a wall.
            2. Take your arm out of the sling, lean forward and allow the affected arm to passively flex with gravity.
            3. Use your other arm to slowly lift the shoulder into flexion.
            4. Ensure that you do not actively use the affected arm.

            suspected complications: hemiplegic shoulder pain (because symptom is pain on movement (active or passive))
            Visit nearby hospital: Yes


        >>Example2<< :

        $Input :
            "diagnosed patient": 'Spinal Cord Injury',
            "patient's disability": 'Activities using arm',
            "functional evaluation": 'CUE-T',
            "ITEMs": ['PINCH DIE', 'WRIST UP'],
            "newly acquired symptoms":'Face looks pale, frequently feel dizzy'

        $output :
            required test : CUE T Test.
            Extract anatomical structures : fingers, wrist.


            Item of which the score dropped : 'Pinch Die'
            Exercise recommendation :
            1. Position yourself with your fingers weaved between each other.
            2. Push up with the fingers of your right hand while pushing down with the fingers of your left hand.

            Item of which the score dropped : 'Wrist up'
            Exercise recommendation :
            1. Position yourself sitting with your hand grasping a cup and hanging over the edge of a table.
            2. Practice tilting the cup up by bending your wrist to a point level with, or higher than the table.

            suspected complications: Orthostatic hypotension (because symptoms are pallor and dizziness)
            Visit nearby hospital: Yes
        ---

        """
    )
    chain = prompt_template | llm | StrOutputParser()

    docs_list = [docs[3], docs[5], docs[6], docs[7]]

    diagnosed_patient = patient_info[0]
    queries = [f"upper extremity, complications"]
    result_context = process_documents(docs_list, queries)

    result = chain.invoke({
        "CUE T Manual": result_context[0],
        "PTX": result_context[1],
        "Stroke Complications": result_context[2],
        "SCI Complications": result_context[3],
        "diagnosed patient": diagnosed_patient,
        "patient's disability": patient_info[0],
        "functional evaluation": patient_info[1],
        "ITEMs": decreased_items,
        "newly acquired symptoms": patient_info[2]
    })

    return result

def rehabilitation_assessment_workflow(llm, docs, file_path):
    # Step 1: Calculate 7-day average
    average_scores = calculate_7day_average(file_path)
    print("7-day average scores calculated.")

    # Step 2: Input current item scores
    print("\nPlease input the current scores for each item:")
    current_scores = input_item_scores()

    # Step 3: Compare scores and identify decreased items
    decreased_items = compare_scores(current_scores, average_scores)
    print("\nItems with decreased scores:", decreased_items)

    # Step 4: Get patient information
    print("\nPlease provide the following patient information:")
    patient_info = get_patient_info()

    result = rehabilitation_evaluation(llm, docs, decreased_items, patient_info)

    print("\nAssessment Result:")
    print(result)

    print("\nWorkflow completed.")
