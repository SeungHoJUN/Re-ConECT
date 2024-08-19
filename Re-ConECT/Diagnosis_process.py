import re
from langchain.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from langchain_core.output_parsers import StrOutputParser

def get_user_input():
    patient_info = {}
    questions = [
        ("patient's chief complaint", "Enter patient's chief complaint: ", lambda x: len(x) > 0),
        ("patient's location", "Enter patient's pain location (e.g. Middle, right): ", lambda x: len(x) > 0),
        ("patient's radiation", "Is there pain radiation? (Yes/No, and location if Yes): ", lambda x: x.lower() in ['yes', 'no'] or (x.lower().startswith('yes') and len(x) > 3)),
        ("patient's severity", "Enter pain severity (mild/moderate/severe): ", lambda x: re.search(r'\b(extremely\s+)?(mild|moderate|severe)\b', x.lower()) is not None),
        ("patient's alleviating factors", "Is pain reduced by lying down? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's pain increase", "Pain increase when looking at (aching/opposite/same) side: ", lambda x: x.lower() in ['aching', 'opposite', 'same']),
        ("patient's numbness or tingling", "Numbness or tingling in arm or hand? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's weakness", "Weaker or thinner arm than before? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's onset of pain", "When did the pain start? ", lambda x: len(x) > 0),
        ("patient's trauma history", "Did pain start within 1 day of trauma? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's lower back pain", "Pain also in lower back? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's morning stiffness", "Stiffness in morning? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's leg symptoms", "Leg weakness or pain? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's coronary heart disease history", "History of coronary heart disease? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's weight loss/appetite", "Weight loss or decreased appetite? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's pregnancy/breastfeeding", "Pregnant or breast feeding? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's prolonged sitting", "Prolonged sitting during work? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's fever", "Fever? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's cancer/steroid history", "History of cancer or steroid use? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's osteoporosis", "Osteoporosis? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's age", "Patient's age: ", lambda x: x.isdigit() and 0 < int(x) < 120),
        ("patient's alcohol/drug use", "Alcoholic or drug abuse? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's HIV status", "HIV? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's leg bending difficulty", "Difficult to bend leg? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's urinary/fecal incontinence", "Urinary or fecal incontinence? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's shoulder drooping or winging", "Shoulder drooping or winging? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's upper neck tenderness", "Tenderness at upper neck? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's arm lift score", "Arm lift against gravity score (0-5): ", lambda x: x.isdigit() and 0 <= int(x) <= 5),
        ("patient's Babinski Reflex", "Babinski Reflex (positive/negative): ", lambda x: x.lower() in ['positive', 'negative']),
        ("patient's sensation in arms", "Sensation difference between arms? (Yes/No): ", lambda x: x.lower() in ['yes', 'no']),
        ("patient's Spurling test", "Spurling test result (positive/negative): ", lambda x: x.lower() in ['positive', 'negative'])
    ]

    total_questions = len(questions)

    for i, (key, question, validator) in enumerate(questions, 1):
        while True:
            answer = input(question)
            if validator(answer):
                patient_info[key] = answer
                progress = (i / total_questions) * 100
                print(f"Progress: {progress:.1f}%")
                break
            else:
                print("Invalid input. Please try again.")

    return patient_info

def create_prompt_template():
    template = """
    You are a renowned rehabilitation medicine specialist. Check the patient's condition and suggest suspected diagnoses, further evaluations, and red flags based on the condition. Reference the provided pain guides during this process.

    1. Check the patient's condition:
       a) Chief complaint: ${patient's chief complaint}
       b) History taking:
          ${history_questions}
       c) Physical examinations:
          ${physical_exam_questions}

    2. Suggest diagnoses, further evaluations, and search for red flags:
       a) Suggest maximum 3 suspected diagnoses based on history and physical examinations.
          Use the <DIFFERENTIAL DIAGNOSIS> section of the {pain guide}. If no match, suggest "unspecified neck pain".
       b) Suggest further examinations based on symptoms, suspected diagnoses, and the <Imaging and Other Diagnostic Tests> section in {pain guide 2}.
       c) Check for red flags:
          - If present: List red flags from the following list and recommend hospital visit:
            * Fever
            * Unexplained weight loss
            * History of cancer
            * History of violent trauma
            * History of steroid use
            * Osteoporosis
            * Aged younger than 20 years or older than 50 years
            * Failure to improve with treatment
            * History of alcohol or drug abuse
            * HIV
            * Lower extremity spasticity
            * Loss of bowel or bladder function
          - If absent: Suggest rehabilitation exercises from PTX guide where the anatomical structure in the chief complaint is included in <Client's aim>.

    Important notes:
    - Maintain patient confidentiality at all times.
    - Provide your output in a structured format as shown in the examples below.

    ${example_outputs}

    Please provide your assessment and recommendations based on the given information.
    """
    return PromptTemplate.from_template(template)

def create_history_questions():
    return "\n".join([
        "- Location (e.g. upper-lower, left-right): ${location}",
        "- Radiation (include radiating location): ${radiation}",
        "- Severity (severe/moderate/mild): ${severity}",
        "- Pain reduced by recumbency (lying down): ${alleviating_factors}",
        "- More painful when looking at aching side vs opposite side vs same: ${pain_increase}",
        "- Numbness or tingling in arm or hand: ${numbness_tingling}",
        "- Weaker or thinner arm than before: ${weakness}",
        "- When did the pain start: ${onset_of_pain}",
        "- Did the pain start within 1 day of a trauma (e.g. traffic accident, lifting): ${trauma_history}",
        "- Pain also in lower back: ${lower_back_pain}",
        "- Stiffness in morning: ${morning_stiffness}",
        "- Leg weakness or pain: ${leg_symptoms}",
        "- History of coronary heart disease: ${coronary_heart_disease_history}",
        "- Weight loss or decreased appetite: ${weight_loss_appetite}",
        "- Pregnant or breast feeding: ${pregnancy_breastfeeding}",
        "- Prolonged sitting during work: ${prolonged_sitting}",
        "- Fever: ${fever}",
        "- History of cancer or steroid use: ${cancer_steroid_history}",
        "- Osteoporosis: ${osteoporosis}",
        "- Age: ${age}",
        "- Alcoholic or drug abuse: ${alcohol_drug_use}",
        "- HIV: ${hiv_status}",
        "- Difficult to bend leg (leg spasticity): ${leg spasticity}",
        "- Urinary or fecal incontinence: ${urinary_fecal_incontinence}"
    ])

def create_physical_exam_questions():
    return "\n".join([
        "- Shoulder drooping or winging: ${shoulder_drooping_winging}",
        "- Tenderness at upper neck: ${upper_neck_tenderness}",
        "- Arm lift against gravity score (0-5): ${arm_lift_score}",
        "- Babinski Reflex (positive/negative): ${babinski_reflex}",
        "- Sensation difference between arms: ${sensation_in_arms}",
        "- Spurling test result (positive/negative): ${spurling_test}"
    ])

def create_example_outputs():
    return """
    >>Example1<<:
    suspected diagnoses:
    1. Infection (evidence: pain not alleviated by lying down)
    2. Cervical radiculopathy (evidence: numbness and tingling, shoulder drooping, arm lift score ≤3, positive Spurling test)
    3. Brachial plexopathy (evidence: numbness and tingling, upper extremity weakness, shoulder and upper extremity pain)

    further examinations: MRI, Electromyography

    finding red flags: Red flags present: "Fever". Urgent hospital visit recommended.

    >>Example2<<:
    suspected diagnoses:
    1. Muscle strain (evidence: pain increased when turning toward opposite side)
    2. Ankylosing spondylitis (evidence: concurrent lower back pain)

    further examinations: Plain Radiography

    finding red flags: Red flags absent.
    Recommended rehabilitation exercise: "Lifting head off two pillows" - Lie on two pillows, slide your head up the pillow, and nod your head. Gently lift your head off the pillow. Perform slowly and controlled, avoiding pain or other symptoms.

    >>Example3<<:
    suspected diagnoses:
    1. Myelopathy (evidence: Babinski reflex positive, leg weakness or pain, numbness or tingling in arm or hand)
    2. Radiculopathy (evidence: more painful when looking at the aching side, numbess or tingling in arm or hand, trauma)
    3. Cancer (evidence: weakness or pain)

    further examinations: Computed tomography, Electromyography

    finding red flags: Red flags present: "Osteoporosis". Urgent hospital visit recommended.
    """

def process_documents(docs_list, queries):
    # This function needs to be implemented based on your specific document processing logic
    # For now, we'll return a dummy result
    return {i: {query: f"Dummy result for document {i}, query: {query}" for query in queries} for i in range(len(docs_list))}

def Diagnosis_Process():
    # Get user input
    patient_data = get_user_input()

    # Create prompt template and other necessary components
    prompt_template = create_prompt_template()
    history_questions = create_history_questions()
    physical_exam_questions = create_physical_exam_questions()
    example_outputs = create_example_outputs()

    # Create the chain
    llm = ChatUpstage(temperature=0)
    chain = prompt_template | llm | StrOutputParser()

    # Process documents
    docs_list = [docs[0], docs[1], docs[2], docs[5]]  # You need to define 'docs' somewhere
    chief_complaint = "neck pain"
    queries = [f"symptoms, findings, diagnoses, evaluations related to {chief_complaint}"]
    result_context = process_documents(docs_list, queries)

    # Invoke the chain
    result = chain.invoke({
        "history_questions": history_questions,
        "physical_exam_questions": physical_exam_questions,
        "example_outputs": example_outputs,
        "pain guide": result_context[0],
        "pain guide 2": result_context[1],
        "pain guide 3": result_context[2],
        "PTX": result_context[3],
        **patient_data
    })

    print(result)