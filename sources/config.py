from os.path import join

RESOURCE_PATH = './resources'
CHECKPOINT_FOLDER = join(RESOURCE_PATH, 'checkpoint')
CHECKPOINT_PATTERN = join(RESOURCE_PATH, '{}_{}.pt')  # model_signature, acc_score

AGE_BINS = [11, 19, 26, 33, 40]
AGE_LABELS = [0, 1, 2, 3]

GENDER_MAPPER = {'male': 1, 'female': 0}
