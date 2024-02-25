import numpy as np 
import random
import copy
import re

class Chromosome:
    def __init__(self, letters):
        
        shuffled = random.sample(letters, 26)     
        letter_mapping = {}
        for original, mapped in zip(letters, shuffled):
            letter_mapping[original] = mapped
        self.key = letter_mapping
        
        self.score = 0

class Population:
    def __init__(self, number_of_chromosomes) -> None:
        self.__letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        self.__population_size = number_of_chromosomes
        self.chromosomes = np.empty(number_of_chromosomes, dtype=Chromosome)

        for i in range(number_of_chromosomes):
            self.chromosomes[i] = Chromosome(self.__letters)
    
    def sort_by_score(self):
        scores = np.array([chromosome.score for chromosome in self.chromosomes])
        sorted_indices = np.argsort(scores)[::-1]
        self.chromosomes = self.chromosomes[sorted_indices]


    def __recombine_two_chromosomes(self, c1: Chromosome, c2:Chromosome, recombination_point:int) -> dict :
        
        #key is our chromosome's mappings. keys are dict keys.
        keys1 = list(c1.key.keys())
        keys2 = list(c2.key.keys())

        # Create a new mapping by combining parts from both mappings
        new_mapping = {}
        new_mapping.update({k: c1.key[k] for k in keys1[:recombination_point]})
        assigned_letters = new_mapping.values()
        unassigned_keys = []
        for key in keys2:
            if key not in new_mapping:
                if c2.key[key] not in assigned_letters:
                    new_mapping[key] = c2.key[key]
                else:
                    if c1.key[key] not in assigned_letters:
                        new_mapping[key] = c1.key[key]
                    else:
                        unassigned_keys.append(key)
        unassigned_values = list(value for value in c2.key.values() if value not in new_mapping.values())
        for key in unassigned_keys:
            if key not in new_mapping:
                value_to_assign = unassigned_values[-1]
                new_mapping[key] = value_to_assign
                unassigned_values.remove(value_to_assign)

        return new_mapping

    def recombination(self, c1: Chromosome, c2:Chromosome):
        recombination_point, _ = self.__generate_swap_points(min_num=7, max_num=20)
        new_mapping_1 = self.__recombine_two_chromosomes(c1, c2, recombination_point)
        new_mapping_2 = self.__recombine_two_chromosomes(c2, c1, recombination_point)
        c1.key = new_mapping_1
        c2.key = new_mapping_2
        return c1, c2
        
    def __generate_swap_points(self, min_num=0, max_num=25):
        mutation_point_1 = random.randint(min_num, max_num)
        mutation_point_2 = mutation_point_1
        while mutation_point_2 == mutation_point_1:
            mutation_point_2 = random.randint(0, 25)
        return mutation_point_1, mutation_point_2
    
    def mutation(self, c, num_of_mutations):
        #key is our chromosome's mappings. keys are dict keys.
        keys = list(c.key.keys())
        for i in range(num_of_mutations):
            mutation_point_1, mutation_point_2 = self.__generate_swap_points()
            c.key[keys[mutation_point_1]], c.key[keys[mutation_point_2]] = c.key[keys[mutation_point_2]], c.key[keys[mutation_point_1]]
        return c

    def generate_new_generation(self):
        self.sort_by_score()
        top_20 = self.chromosomes[0:int(self.__population_size/5)]
        mutated_once = np.empty_like(top_20)
        mutated_twice = np.empty_like(top_20)
        mutated_seven_times = np.empty_like(top_20)
        recombinations = np.empty_like(top_20)
        recombinations_2 = np.empty_like(top_20)
        #recombinations_3 = np.empty_like(top_20)

        for i in range(len(top_20)):
            mutated_once[i] = self.mutation(copy.deepcopy(top_20[i]), 1)
            mutated_twice[i] = self.mutation(copy.deepcopy(top_20[i]), 2)
            mutated_seven_times[i] = self.mutation(copy.deepcopy(top_20[i]), 7)

        for i in range(0, len(top_20), 2):
            recombinations[i], recombinations[i+1] = self.recombination(copy.deepcopy(top_20[i]), copy.deepcopy(top_20[i+1]))
        #    c_1_index, c_2_index = self.__generate_swap_points(min_num=0, max_num=len(top_20)-1)
        #    recombinations_2[i], recombinations_2[i+1] = self.recombination(copy.deepcopy(top_20[c_1_index]), copy.deepcopy(top_20[c_2_index]))
        #    c_1_index, c_2_index = self.__generate_swap_points(min_num=0, max_num=len(top_20)-1)
        #    recombinations_3[i], recombinations_3[i+1] = self.recombination(copy.deepcopy(top_20[c_1_index]), copy.deepcopy(top_20[c_2_index]))
        
        self.chromosomes = np.concatenate([top_20, mutated_once, mutated_twice, recombinations, mutated_seven_times])

class Decoder:
    def __init__(self, encoded_text, global_text) -> None:
        self.encoded_text = encoded_text
        self.global_text = global_text
        
        self.population_size = 320
        self.max_without_improvement = 52
        self.max_number_of_iterations = 800
        self.init_population = Population(self.population_size) 
        self.population = self.init_population

    def __decode_with_key(self, input_text, key) -> str:
        decoded_text = ''
        letter_mapping = {v: k for k, v in key.items()}
        for char in input_text:
            if not char.isalpha():
                decoded_text += char
            else:
                original_letter = char.upper()
                decoded_letter = letter_mapping.get(original_letter, original_letter)
                if not char.isupper():
                    decoded_letter = decoded_letter.lower()
                decoded_text += decoded_letter
        return decoded_text

    def __fitness_func(self, decoded_text, global_text) -> float:
        
        decoded_words = decoded_text.split()
        global_words = global_text.split()

        number_of_shared_words = len(set(decoded_words) & set(global_words))
        
        #Higher is better. Max possible is 1.
        fitness = number_of_shared_words / len(decoded_words)
        return fitness

    def decode(self):
        generation_counter = 0
        best_score = 0.0
        best_key = dict()
        local_optimum_counter = 0
        loop_until_break_condition = True
        while loop_until_break_condition:
            #Calculate score for each chromosome
            print("\nCalculating scores for generation ", generation_counter)
            for i in range(self.population_size):
                decoded_text = self.__decode_with_key(input_text = self.encoded_text, key=self.population.chromosomes[i].key)
                self.population.chromosomes[i].score = self.__fitness_func(decoded_text, self.global_text)
            
            #Sort by Scores
            self.population.sort_by_score()
            new_best_score = self.population.chromosomes[0].score
            print("Best score in generation ", generation_counter, ':', new_best_score)
            best_key = self.population.chromosomes[0].key
            #print(best_key)
            
            if new_best_score == best_score:
                local_optimum_counter += 1
                print("No improvement. Remaining generations: ", self.max_without_improvement - local_optimum_counter)
            else:
                local_optimum_counter = 0
                best_score = new_best_score

            if generation_counter >= self.max_number_of_iterations:
                break
            elif local_optimum_counter == self.max_without_improvement:
                break
            else:
                #Generate new generation
                print("Generating next generation...")
                self.population.generate_new_generation()
                generation_counter += 1
        
        best_key_swapped = dict(sorted({value: key for key, value in best_key.items()}.items()))
        print("\nDecoded letter -> Encoded letter\n")
        for original, mapped in best_key_swapped.items():
            print(f"{original} -> {mapped}")
        return self.__decode_with_key(self.encoded_text, best_key)

def procces_text(input_text:str, add_space=True) -> str:

    if add_space:
        pattern = r'([.,!?;:"\'\(\)\[\]{}<>+=\-_*&^`%$#@])'
        output_text = re.sub(pattern, r' \1 ', input_text)
    else:
        d = { " . ": ". ", " , ": ", ", " ( ": " (", " ) ": ") ",
         " @ ": "@", " ? ": "? ", " ! ": "! ", " ; ": "; ", " : ": ": ",
         " [ ": " [", " ] ": "] ", " ' ": "'", }
        output_text = input_text
        for k,v in d.items():
            output_text = output_text.replace(k, v)
    return output_text

def write_to_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
    except Exception as e:
        print(f'Error writing to file {file_path}: {e}')

def main():
    encoded_text_file = 'Data/encoded_text.txt'
    global_text_file = 'Data/global_text.txt'

    encoded_text, global_text = str(), str()

    with open(encoded_text_file, 'r') as file:
        encoded_text = file.read()

    with open(global_text_file, 'r') as file:
        global_text = file.read()
    
    encoded_text = procces_text(encoded_text, True)
    global_text = procces_text(global_text, True)
    d = Decoder(encoded_text, global_text)
    decoded_text = d.decode()
    decoded_text = procces_text(decoded_text, False)
    print(decoded_text)

    decoded_text_file = 'decoded_text.txt'
    write_to_file(decoded_text_file, decoded_text)

if __name__ == "__main__":
    main()