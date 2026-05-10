This document is for anyone interested in the product vision for chat-workflow. It covers the problem statement, solution, user experience, and philosophy to answer the question: "What is this library trying to achieve and why?"

# Product Specification: Domain-agnostic library allowing the creation of complex structured conversations (flows) that end users can have with LLMs, giving a heavily "human in the loop" approach to designing agents with a deterministic control flow.

## Overview
The "unit" of this library is functions that the flow author creates, using a dedicated decorator, which take and return Pydantic objects whose correctness is verified. The flow author writes the Pydantic types and then authors the functions. Such a function would normally contain a specialised prompt requesting the LLM to facilitate the generation of the object to be returned. Alternatively, the function might compose other such functions using standard Python control flow primitives and libraries to achieve a more complex result. For example, one function might return EvaluationCriteria, another might return a Shortlist of options, and a third function would combine these two and evaluate the Shortlist against the Criteria to generate and return a Decision.

## Problem Statement
Tools to build agent workflows often force workflow authors to build graph structures using visual tools. This is extremely limited compared to the use of program code more directly.
They also aren't great at supporting fully flexible "human in the loop" architectures where the human has most of the knowledge and intuition that the workflow requires, and the workflow functions as a coach/facilitator.
In addition they don't have good support for ensuring the excellence/correctness of all the output.

## Solution
An AI-powered assistant that guides users through an arbitrarily complex structured conversation to define clear, structured end results. For example, a detailed business plan containing a number of distinct sections, each of which requires at least one sub-conversation.
A python library that allows flow authors to build up these conversation flows, define "what good looks like" for each subunit of the conversation (using Pydantic)

## User Experience

### Simple Use Case: Guided Criteria Creation
1. **User starts** with a general idea (e.g., "help me decide on evaluation criteria for birthday presents for a 7-year-old")
2. **AI assistant asks questions** to understand requirements:
   - "What's your budget range?"
   - "Are there any age-appropriate considerations?"
   - "What interests or hobbies should we consider?"
3. **User responds naturally** in conversation
4. **Assistant continues asking** until it has enough information
5. **Assistant produces structured criteria** with:
   - Clear factor names (e.g., "budget", "age_appropriateness", "educational_value")
   - Descriptions of what each factor measures
   - Importance weights (1-10 scale)
   - Optional ideal/target values

### More complex use case: Decision making
1. **User starts** by describing the nature of the decision ("I need to choose a birthday present for my nephew"). There is a "create_decision" function made by the workflow author that returns a Pydantic Decision object that will contain the actual decision the user makes..
2. **Assistant continues** by first facilitating the creation of evaluation criteria as described in the previous example, in a function that returns a Pydantic EvaluationCriteria object. So the create_decision function has had to call a create_evaluation_criteria function to generate the EvaluationCriteria.
4. **Assistant moves on**, facilitating the user to develop a set of options, perhaps including ideas the user already has, perhaps also including options the agent has found on the internet using an MCP for web search. So the create_decision function now calls create_option_shortlist that creates an OptionShortlist Pydantic object.
6. **Assistant continues** in the create_decision function, calling create_assessment and passing the options and the criteria. create_assessment has the LLM evaluate the options against the given criteria and return an Assessment object that shows how each shortlisted option stacks up.
7. **Agent makes a preliminary Decision** based on the Assessment.
8. **Final Decision is generated** by create_decision function, a Decision object is returned and the task is complete.

The LLM's context is always composed from the parameters passed to each function and the prompt added by the function itself. That is, context established prior to the function call is not visible to the LLM unless it's explicitly passed to the function.

### Key Features

#### 1. Natural Conversation Interface
- Chat-based interaction - users talk like they would with a human expert
- AI asks one question at a time to avoid overwhelming users

#### 2. Structured Output Generation
- Each flow function always produces valid, complete Pydantic objects, or else raises an exception
- Output is machine-readable for further analysis or integration

#### 3. Intelligent Limits
Within an individual flow function:
- **Each conversation has a turn limit** - prevents endless loops
- **Three clear outcomes from each LLM call**:
  - ✅ **Success**: Valid object generated
  - 🔄 **Continue**: More questions needed
  - ❌ **Failure**: Can't proceed (user gives up or insufficient info)
- **Business rule enforcement**: Won't produce invalid objects 

#### 4. User-Friendly Error Handling
- Clear explanations when things go wrong
- No technical jargon in error messages
- Graceful failure with actionable suggestions

#### 5. Minimal Boilerplate for Workflow Authors 
The authors of workflows must be able to author these workflows within a dedicated library for that workflow, that only contains the workflow functions themselves. All framework code must be generic, decoupled from the workflows and must be situated in the chat-workflow library. There must be no requirement for the workflow author to copy and amend framework code for each workflow function they write.

## Target Users

### Primary User: Individuals wanting structured professional coaching/facilitation support

- Individuals with complex problems (help me meal plan, help me plan care for my disabled child, help me get my extension built)
- Professionals with complex problems (help me make a business plan, help me manage my time, help me create new brand artifacts)

### Secondary User: Facilitators
- Professionals with specialist knowledge wanting to make and use or resell agents that provide structured coaching/facilitation in their domain
- For example: Architects wanting to streamline the qualification, conversion and onboarding of leads by having potential customers chat to an agent about their ideas.
- For example: Business coaches wanting to sell access to software that can coach you through creating a business plan.

## Usage Scenario

### Generate evaluation criteria 
```
User: "I need a birthday gift for my 7-year-old niece"
Assistant: "I can help with that! What's your budget for this gift?"
User: "Around $50"
Assistant: "Great. Does she have any particular interests or hobbies?"
User: "She loves science and building things"
Assistant: "Perfect. Are there any safety considerations for her age?"
[Conversation continues...]
✅ Success: Generated criteria with factors: budget, educational_value, 
    safety, age_appropriateness, entertainment_value
```

### 2. Debugging Tools 

- Each time a user goes through a flow, we finish by asking the user to evaluate the conversation itself and provide comments. We log these so that the workflow author can improve their workflow over time to be more useful.

### 3. Usability

- We want workflow authors with a knowlege of coding to find it intuitive to build their workflows using this library
- Workflow authors should be able to use commercially available coding agents to build workflows using this library, so the library should be easy to use by coding agents.
- Users should find it easy and friendly to use the resulting workflows. The experience should be comparable to having a chat with a human.

## Success Metrics

### User Success Metrics
- **Completion rate**: Percentage of conversations that produce valid criteria
- **Turn efficiency**: Average turns needed to reach success
- **User satisfaction**: Qualitative feedback on helpfulness

### Technical Success Metrics  
- **API reliability**: Successful LLM interactions vs failures
- **Validation rate**: Percentage of generated criteria that pass business rules
- **Error clarity**: User understanding of error messages

## Integration Points

### Current
- **Command Line Interface**: Primary interaction method
- **Environment configuration**: API key management for different providers

### Future Potential
- We're going to make the workflows self modifying, that is, users will be able to provide feedback in real time and have the workflows update.
- We're likely to use Temporal to make sure users can restart sessions.
- We'll make it so workflows can save critical outputs (objects), maybe as Markdown, maybe into a Minio/formkiq/onyx setup, so that users can start to create a knowledge base around their life/business and continue to use these objects (business plan, meal plan, brand guidlines, etc) as inputs to future workflows.

## Philosophy

### Design Principles
1. **Conversation First**: Natural dialogue over forms or checklists
2. **Structured Second**: Behind-the-scenes structure, front-end simplicity, for both end users and workflow authors.
3. **Fail Helpfully**: Clear explanations, not technical errors

### Technical Philosophy
- **Infrastructure exposure**: Tests fail when API keys missing (exposes setup issues)
- **Real validation**: Comprehensive suite of evals (as well as unit tests) to make sure our code changes work with real LLMs, not just mocks
- **Clean boundaries**: Clear separation of concerns. Low coupling. 

## Why This Matters

This tool makes workflow creation:
- **Accessible**: No expertise in the framework needed
- **Structured**: Results are usable for analysis and comparison
And workflow use is:
- **Thorough**: Systematic semi-deterministic approach ensures nothing is missed  
- **Adaptable**: Works for personal, professional, and team decisions

The conversation format makes what can feel like a dry, analytical task into an engaging, thoughtful dialogue that produces better outcomes.
