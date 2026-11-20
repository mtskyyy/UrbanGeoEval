INIT_PROMPT = '''You are an expert urban geospatial reasoning assistant. Your task is to answer questions with adherence​​ to the following structure and rules.'''

TASK_PROMPT_LIST = {
# distance
  "distance": '''Your response ​​MUST​​ consist of ​​ONLY two sections​​ formatted as follows:
Section 1: ​Reasoning Process​​
Provide a detailed, step-by-step explanation of your reasoning process enclosed within <reason> and </reason> tags. This section should capture all relevant considerations and intermediate steps, ensure that your reasoning is comprehensive yet focused to the question.

Format your reasoning process as outlined below:  
Step 1: Extract Coordinates
 - Clearly state the latitude and longitude of both locations in decimal degrees.  
Step 2: Convert Degrees to Radians
Step 3: Calculate Differences
Step 4: Apply the Haversine Formula
Step 5: Computation

Section 2: Final Answer
Present your concise and definitive final answer enclosed within <answer> and </answer> tags.
​​Format strictly as​​: [Value] [Unit] (e.g., <answer>2.34 km</answer>).''',

# direction
  "direction": '''Your response ​​MUST​​ consist of ​​ONLY two sections​​ formatted as follows:
Section 1: ​Reasoning Process​​
Provide a detailed, step-by-step explanation of your reasoning process enclosed within <reason> and </reason> tags. This section should capture all relevant considerations and intermediate steps, ensure that your reasoning is comprehensive yet focused to the question.

Format your reasoning process as outlined below:  
Step 1: Extract Coordinates
 - Clearly state the latitude and longitude of both locations in decimal degrees.
Step 2: Convert Degrees to Radians
Step 3: Calculate the Difference in Longitude
Step 4: Apply the Bearing Formula
Step 5: Convert the Bearing to Degrees
 - Convert the calculated bearing from radians to degrees.
Step 6: Determine Descriptive Direction
 - Based on the calculated bearing, provide the cardinal or intercardinal direction (e.g., North, Northeast, East-Southeast, etc.)

Section 2: Final Answer
Present your concise and definitive final answer enclosed within <answer> and </answer> tags.
​​Format strictly as​​: Bearing: [Value] degrees\nCardinal direction: [Direction] (e.g., <answer>Bearing: 123.96 degrees\nCardinal direction: Southeast</answer>).''',

# relation
  "relation": '''Your response ​​MUST​​ consist of ​​ONLY two sections​​ formatted as follows:
Section 1: ​Reasoning Process​​
Provide a detailed, step-by-step explanation of your reasoning process enclosed within <reason> and </reason> tags. This section should capture all relevant considerations and intermediate steps, ensure that your reasoning is comprehensive yet focused to the question.

Section 2: Final Answer
Provide your final answer as a ​​single spatial relation term​​ enclosed in <answer> tags. Example: <answer>Intersecting</answer>''',

# planning
  "planning": '''Your response ​​MUST​​ consist of ​​ONLY two sections​​ formatted as follows:
Section 1: ​Reasoning Process​​
Provide a detailed, step-by-step explanation of your reasoning process enclosed within <reason> and </reason> tags. This section should capture all relevant considerations and intermediate steps, ensure that your reasoning is comprehensive yet focused to the question.
Steps to follow:
1. Understand the Problem
2. Gather the Given Data
3. Identify the Farthest Location
4. Plot the Locations
5. Find the Optimal Path

Section 2: Final Answer
Present your concise and definitive final answer enclosed within <answer> and </answer> tags.
​​Format strictly as​​: [Location 1]->[Location 2]->[Location 3]->[Location 4]->[Location 5] (e.g., <answer>Central Park->The Met->Times Square->Statue of Liberty->Brooklyn Bridge</answer>).''',

# prediction
  "prediction": '''Your response ​​MUST​​ consist of ​​ONLY two sections​​ formatted as follows:
Section 1: ​Reasoning Process​​
Provide a detailed, step-by-step explanation of your reasoning process enclosed within <reason> and </reason> tags. This section should capture all relevant considerations and intermediate steps, ensure that your reasoning is comprehensive yet focused to the question.
Steps to follow:
1. Analyze the sequence of locations in the partial trajectory
2. Determine the overall direction of travel
3. Calculate distances between the last recorded position and each candidate destination
4. Consider whether each candidate destination aligns with the established travel pattern
5. Predict the most likely final destination based on directional consistency and geospatial logic

Section 2: Final Answer
Provide only ​​a single location name​​ as your final answer, strictly enclosed within <answer> and </answer> tags. (e.g., <answer>Central Park</answer>)''',

# navigation
  "navigation": '''Your response ​​MUST​​ consist of ​​ONLY two sections​​ formatted as follows:
Section 1: ​Reasoning Process​​
Provide a detailed, step-by-step explanation of your reasoning process enclosed within <reason> and </reason> tags. This section should capture all relevant considerations and intermediate steps, ensure that your reasoning is comprehensive yet focused to the question.

Format your reasoning process as outlined below:  
Step 1: Network Comprehension
Step 2: Route Identification
Step 3: Optimal Route Selection
Step 4: Generate Precise Driving Directions

Section 2: Final Answer
Present your concise and definitive final answer enclosed within <answer> and </answer> tags.''',

# recommendation
  "recommendation": '''Your response ​​MUST​​ consist of ​​ONLY two sections​​ formatted as follows:
Section 1: ​Reasoning Process​​
Provide a detailed, step-by-step explanation of your reasoning process enclosed within <reason> and </reason> tags. This section should capture all relevant considerations and intermediate steps, ensure that your reasoning is comprehensive yet focused to the question.

Section 2: Final Answer
Present your concise and definitive final answer enclosed within <answer> and </answer> tags.'''
}

SCORING_PROMPT = '''You are an impartial and meticulous judge. Your task is to evaluate a model’s response to a question. The response is divided into two parts: <reason> (the model’s reasoning process) and <answer> (the model’s final answer). You will be provided with:
1. The original question
2. The model's response (<reason>...</reason><answer>...</answer>)
3. A checklist that defines the scoring criteria
4. A ground truth final answer for reference in evaluating <answer>

Instructions:
1. Evaluate the <reason> section using the checklist:
2. Analyze how well the reasoning aligns with each criterion in the checklist.
3. Refer to specific points from the checklist in your explanation.
4. Note strengths, weaknesses, and any missing or incorrect reasoning steps.
5. Evaluate the <answer> section by comparing it to the ground truth.
6. Be objective and avoid personal bias.
7. You don't need to output explanations, just end your evaluation with the scores in this exact format:
reason:[[score]], answer:[[score]], total:[[score]]
''' # After your explanation, end your evaluation with the scores in this exact format:

CHECKLISTS = {
  "distance": '''Scoring Rubric – Total: 25 Points
Problem Type: Basic Straight-Line Distance Calculation
Objective: Given two geographic coordinates, calculate the straight-line (great-circle) distance between them in kilometers, showing a step-by-step explanation.
Part 1: Reasoning Process (<reason>) – Max 15 Points
  Coordinate Extraction (2 points)
    2: Clearly and correctly states the latitude and longitude of both locations in decimal degrees.
    1: Minor errors in conversion from degrees-minutes-seconds (DMS) to decimal degrees.
    0: Coordinates are missing, incorrect, or not in decimal format.
  Conversion to Radians (3 points)
    3: Correctly converts all decimal degree values to radians.
    1~2: Minor errors in conversion or no calculaton results given(e.g., wrong precision or wrong formula for one coordinate).
    0: Conversion is missing or significantly incorrect.
  Difference Calculation (3 points)
    3: Accurately calculates the differences in latitude (Δφ) and longitude (Δλ) in radians.
    1~2: Minor mistake in one of the difference values or no calculaton results given.
    0: Differences are missing or clearly incorrect.
  Haversine Formula Application (5 points)
    5: Correctly applies the Haversine formula to compute intermediate values a and c.
    2~3: Minor computation error in a or c or no calculaton results given.
    0: Formula not applied or completely incorrect.
  Clarity & Structure (2 points)
    2: The response is clearly separated into <reason> and <answer> sections and follows a logical structure.
    1: Slight structural confusion or disorganization.
    0: No clear separation or poorly structured response.
Part 2: Final Answer Accuracy (<answer>) – Max 10 Points
  The final answer will be compared against the ground truth distance.
  Scoring Rule: Absolute Error Penalty
    Formula: Score = max(0, 10 - |model_answer - ground_truth|), round the score to two decimal places.
Important Notes:
  The maximum total score is 25 points:
    - Up to 15 points from reasoning (<reason>)
    - Up to 10 points from final answer accuracy (<answer>)
  The minimum total score is 0 points.''',

  "direction": '''Scoring Rubric – Total: 25 Points
Problem Type: Direction Calculation (Bearing between two points)
Objective: Given two geographic coordinates (latitude and longitude of two points of interest), calculate the bearing (angle) from one point to the other, measured clockwise from true north. Additionally, provide the direction in descriptive terms (e.g., Northeast, East-Southeast).
Part 1: Reasoning Process (<reason>) – Max 15 Points
  Coordinate Extraction (2 points)
    2: Clearly and correctly states the latitude and longitude of both locations in decimal degrees.
    1: Minor errors in conversion from degrees-minutes-seconds (DMS) to decimal degrees.
    0: Coordinates are missing, incorrect, or not in decimal format.
  Conversion to Radians (3 points)
    3: Correctly converts all decimal degree values to radians.
    1~2: Minor errors in conversion or no calculation results given (e.g., wrong precision or wrong formula for one coordinate).
    0: Conversion is missing or significantly incorrect.
  Difference Calculation (3 points)
    3: Accurately calculates the differences in longitude (Δlong) in radians.
    1~2: Minor mistake in the difference value or no calculation results given.
    0: Difference is missing or clearly incorrect.
  Bearing Formula Application (5 points)
    5: Correctly applies the bearing formula to compute intermediate values.
    2~3: Minor computation error in the bearing formula or no calculation results given.
    0: Formula not applied or completely incorrect.
  Clarity & Structure (2 points)
    2: The response is clearly separated into and sections and follows a logical structure.
    1: Slight structural confusion or disorganization.
    0: No clear separation or poorly structured response.
Part 2: Final Answer Accuracy (<answer>) – Max 10 Points
  The final answer will be compared against the ground truth bearing and cardinal direction.
  Scoring Rule: Absolute Error Penalty
    Formula: Score = max(0, 10 - (|model answer - ground truth bearing|) * 0.25), round the score to two decimal places.
    If cardinal direction in model answer does not match model answer bearing, Score -2 points, else +0 points.
Important Notes:
  The maximum total score is 25 points:
    - Up to 15 points from reasoning (<reason>)
    - Up to 10 points from final answer accuracy (<answer>)
  The minimum total score is 0 points.''',

  "relation-r2r": '''Scoring Rubric – Total: 25 Points
Problem Type: Road-to-Road Spatial Relationship
Objective: Given two roads with sampled coordinates, determine their spatial relationship (Intersecting, Parallel, or Disconnected).
Part 1: Reasoning Process (<reason>) – Max 15 Points
  Coordinate Extraction (1 points)
    1: Correctly states the coordinates for both roads.
    0: Coordinates are missing or incorrect.
  Intersection/Parallel/Disconnected Criteria (3 points)
    3: Correctly identifies the criteria for determining spatial relations (intersection, parallelism, or disconnection).
    1~2: Minor misunderstanding of the criteria or misinterpretation of road relationships.
    0: Criteria are applied incorrectly or not at all.
  Analysis of Spatial Relationship (8 points)
    8: Correctly analyzes the roads' relationship by considering sampled coordinates, using geometric or mathematical methods such as line intersection analysis, or by comparing directional patterns and slopes of the segments.
    3~7: Minor mistake recognizing road orientations or relation or no computational/spatial reasoning steps are shown.
    0~2: Incorrect analysis or application of incorrect spatial relationships.
  Clarity & Structure (3 points)
    3: The response is clearly separated into <reason> and <answer> sections, logically structured, and easy to follow.
    1~2: Slight structural confusion or lack of clarity.
    0: No clear separation or poorly structured response.
Part 2: Final Answer Accuracy (<answer>) – Max 10 Points
  The final answer will be compared against the ground truth spatial relationship (Intersecting, Parallel, or Disconnected).
  Scoring Rule: 
    Score = 10 if answer is correct else 0.
Important Notes:
  The maximum total score is 25 points:
    - Up to 15 points from reasoning (<reason>)
    - Up to 10 points from final answer accuracy (<answer>)
  The minimum total score is 0 points.''',

  "relation-r2a": '''Scoring Rubric – Total: 25 Points
Problem Type: Road-to-Area Spatial Relationship
Objective: Given a road and an area with coordinates, determine the spatial relationship between them (Pass through the area, Touch the area, or Lie outside the area).
Part 1: Reasoning Process (<reason>) – Max 15 Points
  Coordinate Extraction (1 points)
    1: Correctly states the coordinates for both the road and area.
    0: Coordinates are missing or incorrect.
  Boundary/Path Analysis (3 points)
    3: Correctly identifies the criteria for the spatial relationship based on the position of the road and area (intersection, touching boundary, or lying outside).
    1~2: Minor misunderstanding of the criteria or misinterpretation of road and area interactions.
    0: Criteria are applied incorrectly or not at all.
  Analysis of Spatial Relationship (8 points)
    8: Correctly analyzes the road’s relationship to the area by considering sampled coordinates, using geometric or mathematical methods such as point-in-polygon or line-to-polygon tests.
    3~7: Minor mistake in determining the relationship or no computational/spatial reasoning steps are shown.
    0~2: Incorrect analysis or application of incorrect spatial relationships.
  Clarity & Structure (3 points)
    3: The response is clearly separated into <reason> and <answer> sections, logically structured, and easy to follow.
    1~2: Slight structural confusion or lack of clarity.
    0: No clear separation or poorly structured response.
Part 2: Final Answer Accuracy (<answer>) – Max 10 Points
  The final answer will be compared against the ground truth spatial relationship (Pass through the area, Touch the area, or Lie outside the area).
  Scoring Rule: 
    Score = 10 if answer is correct else 0.
Important Notes:
  The maximum total score is 25 points:
  - Up to 15 points from reasoning (<reason>)
  - Up to 10 points from final answer accuracy (<answer>)
  The minimum total score is 0 points.''',

  "relation-a2a": '''Scoring Rubric – Total: 25 Points
Problem Type: Area-to-Area Spatial Relationship
Objective: Given two areas with boundary coordinates, determine the spatial relationship between them (Overlapping, Adjacent, or Separate).
Part 1: Reasoning Process (<reason>) – Max 15 Points
  Coordinate Extraction (1 points)
    1: Correctly states the coordinates for both areas.
    0: Coordinates are missing or incorrect.
  Overlap/Adjacency/Separation Criteria (3 points)
    3: Correctly identifies the criteria for determining spatial relationships (overlap, adjacency, or separation).
    1~2: Minor misunderstanding of the criteria or misinterpretation of area interactions.
    0: Criteria are applied incorrectly or not at all.
  Analysis of Spatial Relationship (8 points)
    8: Correctly analyzes the areas' relationship by considering boundary coordinates and applying geometric or mathematical methods (e.g., intersection, adjacency check, or area overlap tests).
    3~7: Minor mistake in determining the relationship or no computational/spatial reasoning steps are shown.      
    0~2: Incorrect analysis or application of incorrect spatial relationships.
  Clarity & Structure (3 points)
    3: The response is clearly separated into <reason> and <answer> sections, logically structured, and easy to follow.
    1~2: Slight structural confusion or lack of clarity.
    0: No clear separation or poorly structured response.
Part 2: Final Answer Accuracy (<answer>) – Max 10 Points
  The final answer will be compared against the ground truth spatial relationship (Overlapping, Adjacent, or Separate).
  Scoring Rule: 
    Score = 10 if answer is correct else 0.
Important Notes:
  The maximum total score is 25 points:
  - Up to 15 points from reasoning (<reason>)
  - Up to 10 points from final answer accuracy (<answer>)
  The minimum total score is 0 points.''',

  "planning": '''Scoring Rubric – Total: 30 Points
Problem Type: Itinerary Planning
Objective: Given a set of locations and their relative position description, determine the optimal visiting order to minimize the total straight-line travel distance.
Part 1: Reasoning Process (<reason>) – Max 20 Points
  Coordinate Extraction (2 points)
    2: Correctly extracts the distances and bearings between consecutive locations and organizes them logically.
    1: Minor errors in extracting data or organizing information.
    0: Missing or incorrect data extraction.
  Farthest Location Identification (3 points)
    3: Correctly identifies the farthest location based on direction and distances and logically concludes that this location should be visited last.
    0: Incorrect or missing identification of the farthest location.
  Coordinate Conversion and Calculation (5 points)
    5: Correctly converts bearings and distances to Cartesian coordinates for all locations.
    2~3: Minor mistakes in the conversion or failure to account for the correct bearing angle or distance.
    0~1: Incorrect or missing conversion of bearings to Cartesian coordinates.
  Path Evaluation (8 points)
    8: Correctly consider the reduced search space (e.g., only permutations of locations excluding the farthest one) and correctly identifies and calculates the total distance for the optimal visiting order.
    4~6: Minor mistakes in evaluating paths, or considering unnecessary permutations. 
    0~3: Incorrect or incomplete evaluation of paths.
  Clarity & Structure (2 points)
    2: The response is clearly separated into <reason> and <answer> sections and logically structured.
    1: Slight structural confusion or lack of clarity.
    0: No clear separation or poorly structured response.
Part 2: Final Answer Accuracy (<answer>) – Max 10 Points
  The final answer will be compared against the ground truth optimal visiting order.
  Scoring Rule: 
    The model response is compared to the ground-truth order, with a penalty applied for each incorrect location in the order. If the correct location is not in the correct position, it receives a penalty of 2 points per misplaced location.
Important Notes:
  The maximum total score is 30 points:
    - Up to 20 points from reasoning (<reason>)
    - Up to 10 points from final answer accuracy (<answer>)
  The minimum total score is 0 points.''',

  "prediction": '''Scoring Rubric -- Total: 30 Points
Problem Type：Prediction
Objective: Evaluate an LLM's geospatial reasoning ability by assessing how well it can predict the most likely final destination based on partial trajectory.
Part 1: Reasoning Process (<reason>) -- Max 20 Points
  Trajectory Analysis (5 points)
    5: Correctly maps the trajectory points and identifies the overall direction pattern
    2~3: Minor errors in understanding the trajectory sequence or direction
    0: Fails to properly analyze the trajectory or direction of travel
  Coordinate Processing (4 points)
    4: Accurately processes geographic coordinates to determine spatial relationships
    1~2: Minor errors in coordinate interpretation or spatial calculations
    0: Significant errors in coordinate processing or spatial analysis
  Candidate Destination Evaluation (5 points)
    5: Thoroughly analyzes each candidate destination relative to the trajectory
    2~3: Incomplete analysis of some candidate destinations
    0: Fails to properly evaluate candidates against the trajectory pattern
  Distance Calculation (3 points)
    3: Correctly assesses distances from last position to each candidate
    1: Minor errors in distance assessment
    0: Major errors or omissions in distance calculations
  Logical Reasoning (3 points)
    3: Employs sound geospatial logic to reach a reasoned conclusion
    1~2: Minor flaws in geospatial reasoning
    0: Significant gaps in logical reasoning process
Part 2: Final Answer Accuracy (<answer>) -- Max 10 Points
  Correctness of Prediction (10 points)
    10: Selects the destination that matches the ground truth answer
    1~2: Selects an incorrect destination but very close to the ground truth answer
    0: Selects an incorrect destination that is not close to the ground truth answer
Important Notes:
  The maximum total score is 30 points: 
    - Up to 20 points from reasoning (<reason>)
    - Up to 10 points from final answer accuracy (<answer>)
  The minimum total score is 0 points.''',

  "navigation": '''Scoring Rubric -- Total: 30 Points
Problem Type：Navigation
Objective: Evaluate an LLM's geospatial reasoning ability by assessing how well it can determine the optimal (shortest or fastest) route between two locations given a specific road network.
Part 1: Reasoning Process (<reason>) -- Max 20 Points
  Network Comprehension (5 points)
    5: Correctly identifies all road segments and how they connect to form the complete road network
    2~3: Minor errors in understanding how segments connect
    0: Major misunderstanding of the road network structure
  Route Identification (5 points)
    5: Correctly identifies all possible routes from start to destination
    2~3: Misses some possible routes or includes impossible connections
    0: Fails to identify valid routes or creates non-existent connections
  Distance Calculation (6 points)
    6: Correctly calculates distances for necessary route segments using coordinate data     
    2~4: Minor calculation errors or missing important segements
    0: Major calculation errors or inappropriate method for distance determination
  Optimal Route Selection (2 points)
    2: Correctly identifies the optimal route based on calculated distances
    1: Selects a non-optimal but valid route
    0: Selects an invalid or impossible route
  Clarity & Structure (2 points)
    2: Reasoning is clearly structured and logically flows
    1: Minor organizational issues or unclear explanations
    0: Disorganized or confusing reasoning
Part 2: Final Answer (<answer>) -- Max 10 Points
  The final answer will be compared against the ground truth answer.
  Accuracy (5 points)
    5: Route is completely correct and optimal
    2: Route is valid but not optimal
    0: Route is invalid or impossible
  Completeness (4 points)
    4: Includes all necessary segments in correct order
    1: Minor omissions or ordering issues
    0: Major omissions or incorrect ordering
  Clarity (1 points)
    1: Directions are clear, concise, and human-readable
    0: Directions are confusing or impossible to follow
Important Notes:
  The maximum total score is 30 points:
    - Up to 20 points from reasoning (<reason>)
    - Up to 10 points from final answer accuracy (<answer>)
  The minimum total score is 0 points.''',

  "recommendation-1": '''Scoring Rubric -- Total: 30 Points
Part 1: Reasoning Process (<reason>) -- Max 20 Points
  Route Understanding (4 points)
    4: Correctly interprets the planned route and visualizes the spatial relationship between the route and POIs
    2: Minor errors in understanding the route structure
    0: Fails to comprehend the spatial nature of the route
  Proximity Analysis (5 points)
    5: Accurately assesses each POI's distance/deviation from the planned route
    2~3: Minor errors in proximity assessment
    0: Incorrect proximity analysis or missing evaluation
  Constraint Evaluation (6 points)
    6: Correctly checks operational constraints (opening hours, availability) and thoroughly evaluates user preference constraints (cuisine, ambiance, etc.)
    2~4: Missing or incorrect evaluation of some operational constraints or user preferences 
    0~1: Incomplete or incorrect analysis of user preferences
  Decision Framework (3 points)
    3: Creates a logical framework to weigh trade-offs between proximity and preferences
    0~1: Incomplete decision framework or illogical weighting of factors
  Clarity & Structure (2 points)
    2: Response is clearly organized into logical steps with appropriate reasoning
    1: Slight structural confusion or lack of clarity
    0: Disorganized or unclear reasoning process
Part 2: Final Answer Accuracy (<answer>) -- Max 10 Points
  10: Selects the optimal POI that matches the ground truth answer
  3~5: Selects a reasonable POI that satisfies most constraints but is not optimal
  0: Selects an inappropriate POI that violates key constraints
Important Notes:
  - Proximity to route is typically a primary constraint
  - User preferences must be properly weighted against convenience
  - Time constraints (open hours) often serve as hard disqualifiers
  - The model must correctly interpret spatial relationships to succeed
''',

  "recommendation": '''Scoring Rubric -- Total: 30 Points
Part 1: Reasoning Process (<reason>) -- Max 20 Points
  Location Understanding (4 points)
    4: Correctly identifies the user's location and understands spatial context
    2: Minor errors in interpreting the starting location
    0: Fails to comprehend the spatial nature of the query
  Proximity Analysis (5 points)
    5: Accurately calculates distances between user location and each POI
    2~3: Minor errors in distance calculation
    0: Incorrect proximity analysis or missing evaluation
  Constraint Evaluation (6 points)
    6: Correctly applies distance constraints (e.g., "within X meters") and thoroughly evaluates user preference constraints (ratings, features, etc.)
    2~4: Missing or incorrect application of distance constraints or user preferences
    0~1: Incomplete or incorrect analysis of user preferences
  Decision Framework (3 points)
    3: Creates a logical framework to weigh trade-offs between proximity and preferences
    0~1: Incomplete decision framework or illogical weighting of factors
  Clarity & Structure (2 points)
    2: Response is clearly organized into logical steps with appropriate reasoning
    1: Slight structural confusion or lack of clarity
    0: Disorganized or unclear reasoning process
Part 2: Final Answer Accuracy (<answer>) -- Max 10 Points
  10: Selects the optimal POI that matches the ground truth answer
  3~5: Selects a reasonable POI that satisfies most constraints but is not optimal
  0: Selects an inappropriate POI that violates key constraints''',

  "recommendation-3": '''Scoring Rubric — Total: 30 Points
Part 1: Reasoning Process (<reason>) — Max 20 Points
  Intersection Identification (4 points)
    4: Correctly identifies the intersection point between the two roads
    1~2: Minor errors in determining the intersection point
    0: Fails to properly identify where the roads intersect
  Distance Calculation (5 points)
    4~5: Accurately calculates distances from the intersection to each POI
    2~3: Minor errors in distance calculations or missing calculation
    0: Significant errors in distance measurement methodology or implementation
  Preference Analysis (6 points)
    5~6: Properly interprets user's stated preferences and effectively evaluates each POI against the stated preferences
    0~4: Misinterprets or partially addresses preferences or incomplete evaluation of POIs against preferences 
  Decision Framework (3 points)
    3: Employs a logical framework to balance distance and preference considerations
    0~1: Incomplete framework or improper weighting of factors
  Clarity & Structure (2 points)
    2: Response organized into clear, logical steps with appropriate reasoning
    1: Some structural confusion or lack of clarity
    0: Disorganized or unclear reasoning process
  Part 2: Final Answer Accuracy (<answer>) — Max 10 Points
    10: Selects the optimal POI that matches the ground truth answer
    3~5: Selects a reasonable POI that balances distance and preferences but is not optimal
    0: Selects an inappropriate POI that is either far from the intersection or doesn't match preferences
Important Notes:
  - Proximity to the intersection is typically a primary consideration
  - User preferences must be properly weighted against distance
  - The model must correctly understand and calculate spatial relationships
  - The final answer should be concise and directly respond to the question'''
}