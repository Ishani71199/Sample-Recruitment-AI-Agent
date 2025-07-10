import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict
from IPython.display import Image, display
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod

load_dotenv()
os.environ['CURL_CA_BUNDLE'] = ''

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini", 
    azure_endpoint="https://ai-mbevacloud375524212247.cognitiveservices.azure.com/",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01"
)

# State of the graph
class State(TypedDict):
    application: str
    experience_level: str
    skill_match: str
    response: str


workflow = StateGraph(State)

def categorize_experience(state: State) -> State:
    print("Categorizing the experience level of candidate.")
    prompt = ChatPromptTemplate.from_template(
        "Based on the folloeing job application, categorize as 'Entry-level', 'Mid-level' or 'Senior-level' "
        "Respond with either 'Entry-level', 'Mid-level' or 'Senior-level'"
        "Application: {application}"                
        )
    chain = prompt | llm
    experience_level = chain.invoke({"application":state["application"]}).content
    print(f"Experience Level: {experience_level}")
    return {"experience_level": experience_level}

def assess_skillset(state: State) -> State:
    print("Assessing the skillset of candidate")
    prompt = ChatPromptTemplate.from_template(
        "Based on the job application for a python developer, assess the candidate's skillset"
        "Respond with either 'Match' or 'No Match'"
        "Application:{application}"
        )
    chain = prompt | llm
    skill_match = chain.invoke({"application":state["application"]}).content
    print(f"Skill Match: {skill_match}")
    return {"skill_match": skill_match}

def schedule_hr_interview(state: State) -> State:
  print("Scheduling the interview : ")
  return {"response" : "Candidate has been shortlisted for an HR interview."}

def escalate_to_recruiter(state: State) -> State:
  print("Escalating to recruiter")
  return {"response" : "Candidate has senior-level experience but doesn't match job skills."}

def reject_application(state: State) -> State:
  print("Sending rejecting email")
  return {"response" : "Candidate doesn't meet JD and has been rejected."}

workflow.add_node("categorize_experience", categorize_experience)
workflow.add_node("assess_skillset", assess_skillset)
workflow.add_node("schedule_hr_interview", schedule_hr_interview)
workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
workflow.add_node("reject_application", reject_application)

workflow.add_edge(START, "categorize_experience")
workflow.add_edge("categorize_experience", "assess_skillset")
def route_app(state: State) -> str:
  if(state["skill_match"] == "Match"):
    return "schedule_hr_interview"
  elif(state["experience_level"] == "Senior-level"):
    return "escalate_to_recruiter"
  else:
    return "reject_application"
workflow.add_conditional_edges("assess_skillset", route_app)
workflow.add_edge("escalate_to_recruiter", END)
workflow.add_edge("reject_application", END)
workflow.add_edge("schedule_hr_interview", END)

app = workflow.compile()

with open("graph.md", "w") as f:
    f.write("```mermaid\n")
    f.write(app.get_graph().draw_mermaid())
    f.write("\n```")

def run_candidate_screening(application: str):
  results = app.invoke({"application" : application})
  return {
      "experience_level" : results["experience_level"],
      "skill_match" : results["skill_match"],
      "response" : results["response"]
  }

application_text = "I have 20 years of experience in software engineering with expertise in JAVA"
results = run_candidate_screening(application_text)
print("\n\nComputed Results :")
print(f"Application: {application_text}")
print(f"Experience Level: {results['experience_level']}")
print(f"Skill Match: {results['skill_match']}")
print(f"Response: {results['response']}")