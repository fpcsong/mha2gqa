import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from torch import nn
import json
import argparse
import sys
import random


if torch.cuda.is_available():
    map_location = 'cuda' 
else:
    map_location = 'cpu' 

def calculate_score(matrix, groups):
    score = 0
    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                score += matrix[group[i]][group[j]]
    return score

def swap_random_elements(groups,group_heads):
    group1, group2 = random.sample(range(len(groups)), 2)
    index1, index2 = random.randint(0, group_heads-1), random.randint(0, group_heads-1)
    groups[group1][index1], groups[group2][index2] = groups[group2][index2], groups[group1][index1]

def simulated_annealing(matrix, num_group, max_iter, epoch):
    new_list=[]
    group_heads=matrix.shape[0]//num_group
    best_score = -1000000
    best_groups = []
    for i in range(epoch):
        nodes = list(range(matrix.shape[0]))
        random.shuffle(nodes)
        groups = [nodes[i * group_heads:(i + 1) * group_heads] for i in range(num_group)]
        current_score = calculate_score(matrix, groups)
        if current_score > best_score:
            best_score = current_score
            best_groups = [group[:] for group in groups]

        for iteration in range(max_iter):
            new_groups = [group[:] for group in groups]
            swap_random_elements(new_groups,group_heads)
            new_score = calculate_score(matrix, new_groups)
            new_list.append(new_score)
            if new_score > current_score:
                groups = new_groups
                current_score = new_score

                if new_score > best_score:
                    best_score = new_score
                    best_groups = [group[:] for group in new_groups]
        # print('max pos:'+str(new_list.index(max(new_list))%max_iter))
        # print('max score'+str(best_score))
    ######################################################
    # x_values = list(range(len(new_list)))
    
    # fig, ax = plt.subplots()
    
    # ax.plot(x_values, new_list)
    
    # ax.set_title('max value:'+str(new_list.index(max(new_list))))
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # plt.savefig('/mnt/data/MHA2GQA/line_plot.png')
    ######################################################
    return best_groups, best_score

def calculate_loss(matrixA,matrixB):  
    norm_matrixA=matrixA/matrixA.norm(dim=1,keepdim=True)
    norm_matrixB=matrixB/matrixB.norm(dim=1,keepdim=True)
    similarity=(norm_matrixA*norm_matrixB).sum(1).mean()

    return similarity

def calculate_losses(matrix):  
    n=matrix.shape[0]
    loss=0
    norm_matrix=torch.zeros_like(matrix)
    for i in range(n):
        norm_matrix[i]=matrix[i]/matrix[i].norm(dim=1,keepdim=True)
    for i in range(n):
        for j in range(i):
            loss+=(norm_matrix[i]*norm_matrix[j]).sum(1).mean() 

    return loss

def reorder(best_groups):
    sorted_bestgroup = sorted(best_groups, key=lambda x: sum(x))
    sorted_bestgroup = [sorted(sublist) for sublist in sorted_bestgroup]
    return sorted_bestgroup

def get_new_order(num_group, num_heads, calibration_data):
    similarity_matrix=torch.zeros([num_heads,num_heads])
    # get similarity between every two heads
    for i in range(num_heads):
        for j in range(i):
            if item=='value':
                U, _, Vt = torch.linalg.svd(calibration_data[i].T@calibration_data[j])
                Q = U @ Vt
            else:
                Q=find_transform_K(calibration_data[i],calibration_data[j])

            if standard=='cos':
                mean_sim=calculate_loss(calibration_data[i]@Q,calibration_data[j])
                similarity_matrix[i][j]=mean_sim
                similarity_matrix[j][i]=mean_sim
            else:#dist
                mean_dim=((calibration_data[i]@Q-calibration_data[j])**2).sum(1).mean()    
                similarity_matrix[i][j]=-mean_dim                                
                similarity_matrix[j][i]=-mean_dim
    
    max_iter = 3000
    epoch=200
    best_groups, best_score = simulated_annealing(similarity_matrix,num_group, max_iter,epoch)
    best_groups=reorder(best_groups) #just to decrease calculation error
    print("best groups:", best_groups)
    print("best score:", best_score)

    return best_groups

def generalized_procrustes_analysis(matrices, V, O, tolerance=1e-5, max_iterations=100):
    n_matrices, n_points, n_dimensions = matrices.shape
    mean_shape = matrices[0]

    transformed_matrices = torch.zeros_like(matrices)
    Q_memo=torch.zeros((n_matrices,n_dimensions,n_dimensions), device=map_location).to(torch.float64)
    for i, matrix in enumerate(matrices):
        if i==0:
            transformed_matrices[i] = matrix
            Q_memo[i]=torch.eye(n_dimensions).to(torch.float64)
            continue
        U, _, Vt = torch.linalg.svd(matrix.T @ mean_shape)
        Q = U @ Vt
        transformed_matrices[i] = matrix@Q
        Q_memo[i]=Q
    
    if n_matrices==2:
        for i in range(n_matrices):
            V[i]=Q_memo[i].T@V[i]
            O[i]=O[i]@Q_memo[i]
        return V,O

    loss=calculate_losses(transformed_matrices)

    mean_shape= torch.mean(transformed_matrices, axis=0)
    for iteration in range(max_iterations):
        transformed_matrices = torch.zeros_like(matrices)
        Q_memo=torch.zeros((n_matrices,n_dimensions,n_dimensions), device=map_location).to(torch.float64)
        for i, matrix in enumerate(matrices):
            U, _, Vt = torch.linalg.svd(matrix.T @ mean_shape)
            Q = U @ Vt
            transformed_matrices[i] = matrix@Q
            Q_memo[i]=Q
                   
        new_mean_shape = torch.mean(transformed_matrices, axis=0)
        new_loss=calculate_losses(transformed_matrices)

        if new_loss-loss < tolerance:
            break
        mean_shape = new_mean_shape
        #print(loss)
        loss=new_loss
    #print(new_loss)  

    for i in range(n_matrices):        
        V[i]=Q_memo[i].T@V[i]
        O[i]=O[i]@Q_memo[i]
        
    return V,O


def find_transform_K(matrix, mean_shape):
    R_size = matrix.shape[1]
    half_size=R_size//2
    R = torch.zeros((R_size, R_size), device=map_location).to(torch.float64)
    for i in range(half_size):
        U, _, Vt  = torch.linalg.svd(matrix[:,[i,i+half_size]].T@ mean_shape[:,[i,i+half_size]])
        Q = U @ Vt

        if torch.det(Q) < 0:
            U[:, -1] *= -1
            Q = U@Vt
        
        R[i,i]=Q[0][0]
        R[i,i+half_size]=Q[0][1]
        R[i+half_size,i]=Q[1][0]
        R[i+half_size,i+half_size]=Q[1][1]
    return R


def rotate_matrix_QK(matrices, Q, K, tolerance=1e-5, max_iterations=100):
    n_matrices, n_points, n_dimensions = matrices.shape
    mean_shape = matrices[0]

    R_memo=torch.zeros((n_matrices,n_dimensions,n_dimensions), device=map_location).to(torch.float64)

    transformed_matrices = torch.zeros_like(matrices)
    for i, matrix in enumerate(matrices):
        if i==0:
            transformed_matrices[i] = matrix
            R_memo[i] = torch.eye(n_dimensions).to(torch.float64)
            continue
        R_memo[i] = find_transform_K(matrix, mean_shape)
        transformed_matrices[i] = matrix@R_memo[i]

    if n_matrices==2:
        for i in range(n_matrices):
            Q[i]=R_memo[i].T@Q[i]
            K[i]=R_memo[i].T@K[i]
        return Q, K

    loss=calculate_losses(transformed_matrices)
    mean_shape= torch.mean(transformed_matrices, axis=0)

    for iteration in range(max_iterations):
        transformed_matrices = torch.zeros_like(matrices)
        R_memo=torch.zeros((n_matrices,n_dimensions,n_dimensions), device=map_location).to(torch.float64)
        for i, matrix in enumerate(matrices):
            R = find_transform_K(matrix, mean_shape)
            transformed_matrices[i] = matrix@R
            R_memo[i]=R
                 
        new_mean_shape = torch.mean(transformed_matrices, axis=0)

        new_loss=calculate_losses(transformed_matrices)
        if new_loss-loss < tolerance:
            break
        
        mean_shape = new_mean_shape
        #print(loss)    
        loss=new_loss
    #print(new_loss)        

    for i in range(n_matrices):
        Q[i]= R_memo[i].T@Q[i]
        K[i]= R_memo[i].T@K[i]

    return Q, K


def rearrange_VO(num_group, head_dim, value_weight, output_weight, calibration_data):
    hidden_size=value_weight.shape[-1]
    value_weight = value_weight.view(num_group, -1, head_dim, hidden_size).to(torch.float64)
    output_weight = output_weight.T.reshape(num_group,-1, head_dim, hidden_size).transpose(-1, -2).to(torch.float64)
    re_calibration_data=calibration_data.view(num_group,-1,calibration_data.shape[1],calibration_data.shape[2])
    new_value_weight = torch.zeros_like(value_weight)
    new_output_weight =torch.zeros_like(output_weight)
    for i in range(num_group):
        #print("-------------------------------------------------------------")
        new_value_weight[i],new_output_weight[i]=generalized_procrustes_analysis(re_calibration_data[i],value_weight[i],output_weight[i])

    
    new_value_weight = new_value_weight.reshape(-1, hidden_size)
    new_output_weight = new_output_weight.transpose(-1, -2).reshape(-1, hidden_size).T.contiguous()
    return new_value_weight, new_output_weight


def rearrange_QK(num_group, head_dim, query_weight, key_weight, calibration_data):
    hidden_size=key_weight.shape[-1]
    key_weight = key_weight.view(num_group, -1, head_dim, hidden_size).to(torch.float64)
    query_weight = query_weight.view(num_group, -1, head_dim, hidden_size).to(torch.float64)
    re_calibration_data=calibration_data.view(num_group,-1,calibration_data.shape[1],calibration_data.shape[2])
    new_key_weight = torch.zeros_like(key_weight)
    new_query_weight =torch.zeros_like(query_weight)
    for i in range(num_group):
        #print("-------------------------------------------------------------")
        new_query_weight[i], new_key_weight[i] = rotate_matrix_QK(re_calibration_data[i],query_weight[i], key_weight[i])

    new_query_weight=new_query_weight.reshape(-1, hidden_size)
    new_key_weight=new_key_weight.reshape(-1, hidden_size)

    return new_query_weight, new_key_weight

def rearrange_model(model_class,num_group,new_model_path,order_file=None):
    model=model_class.model
    num_heads = model.config.num_attention_heads
    hidden_size=model.config.hidden_size
    head_dim=hidden_size//num_heads
    best_groups=[]
    if order_file:
        with open(order_file, 'r') as f:
            best_groups = json.load(f)
    print("matrix transformation begins!")
    for idx, decoder_layer in enumerate(model.layers):

        query_weight = decoder_layer.self_attn.q_proj.weight.clone()
        key_weight = decoder_layer.self_attn.k_proj.weight.clone()
        value_weight = decoder_layer.self_attn.v_proj.weight.clone()
        output_weight = decoder_layer.self_attn.o_proj.weight.clone()
        
        calibration_value=torch.load(args.calibration_data_path+'/layer'+str(idx)+'-value.pt', map_location=map_location).to(torch.float64)
        calibration_key=torch.load(args.calibration_data_path+'/layer'+str(idx)+'-key.pt', map_location=map_location).to(torch.float64)
        if standard=='cos':#normalization
            print("use_norm!")
            calibration_value=calibration_value/calibration_value.norm(dim=-1,keepdim=True)
            calibration_key=calibration_key/calibration_key.norm(dim=-1,keepdim=True)
        
        if item in ['key','value']:
            if not order_file:
                best_group=[]
                if item=='key':
                    best_group = get_new_order(num_group,num_heads, calibration_key)
                else:
                    best_group = get_new_order(num_group,num_heads, calibration_value)
                new_order = sum(best_group, [])
                best_groups.append(best_group)
                #print(best_group)
            else:
                best_group=best_groups[idx]
                best_group=reorder(best_group)
                best_groups[idx]=best_group
                new_order = sum(best_group, [])
      
            query_weight=torch.cat([query_weight[i*head_dim:(i+1)*head_dim, :] for i in new_order], dim=0)
            key_weight=torch.cat([key_weight[i*head_dim:(i+1)*head_dim, :] for i in new_order], dim=0)
            value_weight=torch.cat([value_weight[i*head_dim:(i+1)*head_dim, :] for i in new_order], dim=0)
            output_weight=torch.cat([output_weight[:,i*head_dim:(i+1)*head_dim] for i in new_order], dim=1)
            calibration_value=torch.cat([calibration_value[i].unsqueeze(0) for i in new_order],dim=0)
            calibration_key=torch.cat([calibration_key[i].unsqueeze(0) for i in new_order],dim=0)
        
        #orthogonal transformations
        new_value_weight, new_output_weight = rearrange_VO(num_group,head_dim,value_weight,output_weight, calibration_value)  #transform W_V and W_O
        new_query_weight, new_key_weight = rearrange_QK(num_group,head_dim,query_weight,key_weight, calibration_key)  #transform W_Q and W_K
            

        decoder_layer.self_attn.q_proj.weight=torch.nn.Parameter(new_query_weight.to(torch.float32))
        decoder_layer.self_attn.k_proj.weight=torch.nn.Parameter(new_key_weight.to(torch.float32))
        decoder_layer.self_attn.v_proj.weight=torch.nn.Parameter(new_value_weight.to(torch.float32))
        decoder_layer.self_attn.o_proj.weight=torch.nn.Parameter(new_output_weight.to(torch.float32))
        print("transformation of layer"+str(idx)+" is done")
    model_class.save_pretrained(new_model_path)
    # save new order to the file

    with open(new_model_path+'/my_list.json', 'w') as f:
        json.dump(best_groups, f)
    print("matrix transformation ends!")

parser = argparse.ArgumentParser(description='model transformation')
parser.add_argument('--model_base', type=str, default='LLaMA-1.3B', help='base model name')
parser.add_argument('--model_path', type=str, default='/workspace/Sheared-LLaMA-1.3B', help='original model path')
parser.add_argument('--calibration_data_path', type=str, default='/workspace/mha2gqa/calibration_data', help='original model path')
parser.add_argument('--output_model_path', type=str, default='/workspace',help='output model path')
parser.add_argument('--group_criterion', type=str, default='dist', help='dist or cos')
parser.add_argument('--group_num', type=int, default=8, help='group number of GQA')
parser.add_argument('--item', type=str,default='value', help='key, value or none')
parser.add_argument('--order_file', type=str,default=None, help='if you already have an order file')
args = parser.parse_args()

standard=args.group_criterion#'cos'# ''dist'#
group=args.group_num  #'8'#'16'# 
order_file=args.order_file
item=args.item #'value'#'none'# or 'key'#

model_path = args.model_path 
new_model_path=args.output_model_path+"/"+args.model_base+"-groups"+str(group)+'-'+standard+'-'+item

model = AutoModelForCausalLM.from_pretrained(model_path).to(map_location)
print(new_model_path)
rearrange_model(model, int(group), new_model_path,order_file=order_file)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(new_model_path)

print('completed.')