import numpy as np

def viterbi_alignment_mod(seq1, seq2, trans_probs, emit_probs):
    length1, length2 = len(seq1), len(seq2)
    state_labels = ['MATCH', 'INSERT', 'DELETE']
    state_count = len(state_labels)

    score_matrix = np.full((length1 + 1, length2 + 1, state_count), float('-inf'))
    path_matrix = np.zeros((length1 + 1, length2 + 1, state_count), dtype=int)

    score_matrix[0][0][0] = 1.0

    for x in range(1, length1 + 1):
        for y in range(1, length2 + 1):
            for current_state in range(state_count):
                for prev_state in range(state_count):
                    transition_probability = trans_probs[prev_state][current_state]
                    current_score = score_matrix[x-1][y-1][prev_state] * transition_probability

                    if current_state == 0:
                        current_score *= emit_probs[current_state][char_to_index(seq1[x-1])]
                        if seq1[x-1] != seq2[y-1]:
                            current_score = float('-inf')
                    else:
                        current_score *= emit_probs[current_state][char_to_index(seq1[x-1])]

                    if current_score > score_matrix[x][y][current_state]:
                        score_matrix[x][y][current_state] = current_score
                        path_matrix[x][y][current_state] = prev_state

    alignment_seq1, alignment_seq2 = "", ""
    state_path = []
    x, y, current_state = length1, length2, np.argmax(score_matrix[length1][length2])

    while x > 0 and y > 0:
        state_path.append(state_labels[current_state])
        if current_state == 0:
            alignment_seq1 = seq1[x-1] + alignment_seq1
            alignment_seq2 = seq2[y-1] + alignment_seq2
            x -= 1
            y -= 1
        elif current_state == 1:
            alignment_seq1 = "-" + alignment_seq1
            alignment_seq2 = seq2[y-1] + alignment_seq2
            y -= 1
        else:
            alignment_seq1 = seq1[x-1] + alignment_seq1
            alignment_seq2 = "-" + alignment_seq2
            x -= 1
        current_state = path_matrix[x][y][current_state]

    return list(reversed(state_path)), alignment_seq1, alignment_seq2

def char_to_index(char):
    return {'A': 0, 'D': 1}.get(char, -1)

trans_probs = [
    [0.9, 0.05, 0.05],
    [0.45, 0.1, 0.45],
    [0.45, 0.45, 0.1]
]

emit_probs = [
    [0.6, 0.4],
    [0.5, 0.5],
    [0.5, 0.5]
]

seq1, seq2 = "ADADA", "AADDD"

viterbi_path, aligned_s1, aligned_s2 = viterbi_alignment_mod(seq1, seq2, trans_probs, emit_probs)

print("Viterbi Path: ", viterbi_path)
print("Aligned Sequences: ")
print("Sequence 1: ", aligned_s1)
print("Sequence 2: ", aligned_s2)
