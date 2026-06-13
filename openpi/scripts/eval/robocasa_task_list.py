"""
RoboCasa v1.0 task registry (target split).

Atomic tasks are single-stage; composite tasks chain two or more atomics.
Each entry provides the robosuite env name and a canonical language prompt.
The prompt is a fallback — at runtime, env.get_ep_meta()["lang"] is preferred.
"""

# fmt: off

# ------------------------------------------------------------------
# Atomic tasks  (v1.0 target/atomic/)
# ------------------------------------------------------------------
ATOMIC_TASKS = {
    "CloseBlenderLid":          "close the lid blender by securely placing the lid on top",
    "CloseFridge":              "close the fridge door",
    "CloseToasterOvenDoor":     "close the toaster oven door",
    "CoffeeSetupMug":           "pick the mug from the counter and place it under the coffee machine dispenser",
    "OpenCabinet":              "open the cabinet door",
    "OpenDrawer":               "open the drawer",
    "OpenStandMixerHead":       "open the stand mixer head",
    "PickPlaceCounterToCabinet":"pick the object from the counter and place it in the cabinet",
    "PickPlaceCounterToStove":  "pick the object from the counter and place it on the stove",
    "PickPlaceDrawerToCounter": "pick the object from the drawer and place it on the counter",
    "PickPlaceSinkToCounter":   "pick the object from the sink and place it on the counter",
    "PickPlaceToasterToCounter":"place the toasted item on a plate",
    "SlideDishwasherRack":      "fully slide the dishwasher rack in",
    "TurnOffStove":             "turn off the stove burner",
    "TurnOnElectricKettle":     "press down the lever to turn on the electric kettle",
    "TurnOnMicrowave":          "press the start button on the microwave",
    "TurnOnSinkFaucet":         "turn on the sink faucet",
}

# ------------------------------------------------------------------
# Composite tasks  (v1.0 target/composite/)
# ------------------------------------------------------------------
COMPOSITE_TASKS = {
    "AddIceCubes":              "Put the ice cubes in the blender.",
    "BreadAndCheese":           "Place the bread and cheese on the cutting board.",
    "CerealAndBowl":            "Open the cabinet. Pick the cereal and bowl from the cabinet and place them on the counter next to the milk. Then close the cabinet.",
    "ChooseMeasuringCup":       "Open the left drawer and place the smaller measuring cup on the counter.",
    "CondimentCollection":      "Pick the two condiments from the counter and place them in the cabinet.",
    "CupcakeCleanup":           "Move the fresh-baked cupcake off the tray onto the counter, and place the bowl used for mixing into the sink.",
    "DessertUpgrade":           "Move the dessert items from the plates to the tray.",
    "DryDishes":                "Pick the cup and bowl from the sink and place them on the counter for drying.",
    "GatherVegetables":         "Pick up the two vegetables from the sink and place them on the cutting board on the counter.",
    "OrganizeCoffeeCondiments": "Pick the two items from the cabinet and place them next to the mug.",
    "RecycleStackedYogurt":     "Stack the yogurt containers by placing the smaller containers on top of the larger containers.",
    "RemoveCuttingBoardItems":  "Remove the vegetable and meat from the cutting board to prepare it for cleaning.",
    "FryingPanAdjustment":      "Pick and place the pan from the current burner to another burner and turn the burner on.",
    "SnackSorting":             "Place the bar in the bowl and close the drawer.",
    "StartElectricKettle":      "Close the electric kettle lid and turn on the electric kettle.",
    "ThawInSink":               "Pick the orange from the counter and place it in the sink. Then turn on the sink faucet.",
    "ArrangeBreadBasket":       "open the cabinet, pick up the bread from the cabinet and place it in the basket, then move the basket to the dining counter",
    "ArrangeTea":               "pick the kettle from the counter and place it on the tray, then pick the mug from the cabinet and place it on the tray, then close the cabinet doors",
    "BreadSelection":           "select a croissant and place it on the cutting board, then retrieve a jar of jam from the cabinet and place it alongside the croissant",
    "CategorizeCondiments":     "put the shaker and condiment bottle from the counter next to their counterparts in the cabinet",
    "CuttingToolSelection":     "place the appropriate cutting tool on the cutting board",
    "DeliverStraw":             "take a straw from the drawer and place it inside the glass cup on the dining counter",
    "GarnishPancake":           "take the strawberry from the fridge and place it on top of the pancake on the dining counter",
    "GatherTableware":          "gather all objects into one cabinet and sort the glasses and bowls to opposite sides",
    "GetToastedBread":          "start the toaster, once the lever pops up, take the bread to the plate on the dining counter",
    "HeatKebabSandwich":        "place the kebab skewer and baguette bread inside the toaster oven, close the door and set the timer",
    "KettleBoiling":            "pick the kettle from the counter, place it on a stove burner, then turn the burner on",
    "LoadDishwasher":           "pick up the cup and bowl from the counter, place them in the dishwasher, and close the dishwasher door",
    "MakeIceLemonade":          "grab a lemon wedge from the fridge and one ice cube from the ice bowl, and put them in the glass of lemonade",
    "PackIdenticalLunches":     "place one eggplant and one chicken drumstick in each tupperware to pack two identical lunches",
    "PanTransfer":              "pick up the pan and dump the vegetables onto the plate, then return the pan to the stove",
    "PortionHotDogs":           "place one bun and one sausage from the bowl on each plate",
    "PreSoakPan":               "pick the pan and sponge and place them into the sink, then turn on the water",
    "PrepareCoffee":            "pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button",
    "RecycleBottlesByType":     "move the plastic bottles to the plastics group and the glass bottles to the glass group",
    "RinseSinkBasin":           "turn on the sink and maneuver the spout to wash all locations of the sink basin",
    "ScrubCuttingBoard":        "pick up the sponge and scrub the cutting board, then release the sponge",
    "SearingMeat":              "grab the pan from the cabinet, place it on the stove burner, place the meat on the pan, then turn the burner on",
    "SeparateFreezerRack":      "place the shrimp container on the second highest freezer rack and the vegetable container on the highest rack",
    "SetUpCuttingStation":      "pick up the knife from the drawer and place it on the cutting board, then place the meat on the cutting board",
    "StackBowlsCabinet":        "pick up the bowls on the counter and stack them in the cabinet with the smaller bowl on top",
    "SteamInMicrowave":         "pick the onion from the sink, place it in the bowl, put the bowl in the microwave, close the door and press start",
    "StirVegetables":           "put the vegetables in the pot, then retrieve the spatula and stir the vegetables",
    "StoreLeftoversInBowl":     "pick the food items from their plates and place them in the bowl, then put the bowl in the fridge",
    "WaffleReheat":             "open the microwave, place the bowl with waffle inside, close the door and turn it on",
    "WashFruitColander":        "put the colander in the sink, place the fruit in the colander, turn on the faucet",
    "WashLettuce":              "wash the lettuce in the sink by running water over it",
    "WeighIngredients":         "pick the jam and place it on the digital scale for weighing, then close the cabinet",
}

ALL_TASKS = {**ATOMIC_TASKS, **COMPOSITE_TASKS}

ATOMIC_TASK_NAMES    = list(ATOMIC_TASKS.keys())
COMPOSITE_TASK_NAMES = list(COMPOSITE_TASKS.keys())
ALL_TASK_NAMES       = list(ALL_TASKS.keys())
