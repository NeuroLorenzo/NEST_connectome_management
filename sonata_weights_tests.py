import nest
from pathlib import Path
import os, sys
import json
import h5py
import time
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz, vstack

nest.set_verbosity("M_ALL")
nest.ResetKernel()
dt=.1
nest.SetKernelStatus({""
                        "local_num_threads": 4,
                        "data_path": 'output/',
                        "print_time" : True,
	                    "rng_seed": 12345,
	                    "resolution": dt
                        
                        })
###############################################################################
# net_par =json.load(open('network_params.json','r'))
sonata_folder='SONATA_COMMUNITY_copia'
with open(sonata_folder+'/config.circuit.json', 'r') as file:
    config_f = json.load(file)
    config_f['run']['tstop'] = 10
    # config_f['run']['tstop'] = net_par['Simulation']['tstop']
with open(sonata_folder+'/config.circuit.json', 'w') as f:
    json.dump(config_f , f, indent=4)


base_path = Path(__file__).resolve().parent
sonata_path = base_path / "SONATA_COMMUNITY_copia"
net_config = sonata_path / "config.circuit.json"


module_name='target/ht_tso_stdp_module' 
# module_name='../target/ht_stdp_tso_sym_module'
# neuron_model_name='hill_tononi_neuron_nestml__with_stdp_tsodyks_Tartarini_synapse_nestml'
# synapse_model_name='stdp_tsodyks_Tartarini_synapse_nestml__with_hill_tononi_neuron_nestml'

nest.Install(module_name)




###############################################################################
print('Start config...')
t = time.time()

sonata_net = nest.SonataNetwork(net_config)
print('Done in ',int(time.time()-t), 's.')

# :py:meth:`~.SonataNetwork.BuildNetwork()` returns a dictionary containing
# the created :py:class:`.NodeCollection`\s. The population names are the
# dictionary keys.

print('Start build...')
t = time.time()
node_collections = sonata_net.BuildNetwork(hdf5_hyperslab_size=2**20)
print('Done in ',int(time.time()-t), 's.')

###############################################################################
# We can now verify whether the built network has the expected number of
# nodes and connections.

print(f'Network built from SONATA specifications in directory "{sonata_path.name}"')
print(f"  Number of nodes      : {nest.network_size}")
print(f"  Number of connections: {nest.num_connections}")

###############################################################################

# RECORDING

# record from CA3
s_rec = nest.Create("spike_recorder")
s_rec.record_to = 'ascii'
s_rec.label = "same_bg_CA3_comm_Network"
pop_name = "CA3_comm_Network"

nest.Connect(node_collections[pop_name], s_rec)
##nest.Connect(node_collections[pop_name][record_node_ids], s_rec)



### add background
pop_name = "CA3_comm_Network"






# nest.SetKernelStatus({"data_path": "output", "overwrite_files":True})


bunch_nodes = pd.read_csv('data/stim_gid_400.csv',header=None)[0].tolist()
bunch_nodes.sort()


# #1
# retrieval_stim1 = nest.Create('spike_generator', 1 )
# stim_site1 = nest.Create('parrot_neuron',1) 
# ret_spk = np.array([2.0,7.0,12.0,17.0]) + 1000
# nest.Connect(retrieval_stim1, stim_site1)
# nest.SetStatus(retrieval_stim1, {"spike_times": ret_spk} )
# print(ret_spk)
# nest.Connect(stim_site1, node_collections[pop_name][bunch_nodes[:20]], syn_spec={'synapse_model':'static_synapse','receptor_type':1,'weight':3.})
# # nest.Connect(stim_site1, bunch_nodes[:20], syn_spec={'synapse_model':'static_synapse','receptor_type':1,'weight':3.})


# pre = np.load('data/weights/pre_t70_l00005.npy')
# post = np.load('data/weights/post_t70_l00005.npy')
# ww = np.load('data/weights/weight_t70_l00005.npy')


# id_pyr = []
# with h5py.File(placement_file,'r') as positions:
#     id_pyr = np.append(id_pyr, positions['SP_PC'][:,0])
#     id_pyr = np.append(id_pyr, positions['SO_PC'][:,0])

# for idp in (id_pyr.astype(int) + 1):
#     #print(idp)
#     conn = nest.GetConnections(node_collections[pop_name][idp-1])
#     source = nest.GetStatus(conn, 'source')
#     targ = nest.GetStatus(conn, 'target')
#     mask = (pre==idp)
#     saved_targ = post[mask]
#     index_mapping ={}
#     for index,value in enumerate(targ):
#         if value in index_mapping:
#             index_mapping[value].append(index)
#         else:
#             index_mapping[value] = [index]

#     found_index = np.empty_like(saved_targ, dtype=int)
#     for i, value in enumerate(saved_targ):
#         idx = index_mapping[value].pop(0)
#         found_index[idx] = i

#     #print(idp,source[0], len(targ),len(ww[pre==idp]), len(found_index) )
#     conn.set({'weight':ww[mask][found_index]})



# dict_neurons={}
# for neuron in node_collections[pop_name]:
# 	# print(neuron) 
# 	dict_neurons[nest.GetStatus(neuron,'global_id')[0]]=[nest.GetStatus(neuron,'tau_m'),len(nest.GetConnections(neuron))]
# # the json file where the output must be stored
# out_file = open("network_sonata.json", "w")

# json.dump(dict_neurons, out_file, indent = 6)

# out_file.close()


# # record from INPUT
i_rec = nest.Create("spike_recorder")
i_rec.record_to = 'ascii'
i_rec.label = "Stimulus"
pop_name = "INPUT_Network"
record_node_ids = [0]
nest.Connect(node_collections[pop_name], i_rec)

# if net_par["Modules"]["SIMULATE"]:
#     print('Start simulate...')
#     print('Simulation time: ', net_par['Simulation']['tstop'], ' ms')
#     t = time.time()
#     sonata_net.Simulate()

pop_name = "CA3_comm_Network"
sonata_folder='SONATA_COMMUNITY_copia'

nodes_csv_filename=sonata_folder+'/network/CA3_comm_Network_node_types.csv'
df=pd.read_csv(nodes_csv_filename, delim_whitespace=True)
print(df.columns.tolist(),flush=True)
pyr_node_ids=[]
for i,pop_name in enumerate(df['pop_name']):
    if pop_name.count('PC'):
        print(i, pop_name, df['node_type_id'][i])
        pyr_node_ids.append(df['node_type_id'][i])
print('pyr node ids',pyr_node_ids,flush=True)
pyr_neuron_names=['SO_PC','SP_PC']

start_time=time.time()
with h5py.File(sonata_folder+'/network/CA3_comm_Network_nodes.h5') as f:
    print(f['nodes/CA3_comm_Network/'].keys(),flush=True)
    # print(f['nodes/CA3_comm_Network/node_group_id'][...])
    group_ids=np.array(f['nodes/CA3_comm_Network/node_type_id'])
    mask=np.isin(group_ids,pyr_node_ids)
    pyr_gids=f['nodes/CA3_comm_Network/node_id'][mask].flatten()
    print('pyr gids of shape:', pyr_gids.shape,flush=True)
    # print(f['nodes/CA3_comm_Network/node_type_id'][...])
pop_name = "CA3_comm_Network"

# batch_size = 200
# batches = [pyr_gids[i:np.min([i+batch_size,len(pyr_gids)])] for i in range(0, len(pyr_gids), batch_size)]
# for i,batch in enumerate(batches):
#     pyr_nodes = node_collections[pop_name][batch]
#     print('querytime =' ,time.time()-start_time)
#     start_time=time.time()
#     conns = nest.GetConnections(source=pyr_nodes).get(
#                 ("source", "target", "weight"), output="pandas")
#     print('querytime =' ,time.time()-start_time)
#     start_time=time.time()

#     import pickle
#     with open(f'conn_{i}.pkl', "wb") as f:
#         pickle.dump(conns, f, pickle.HIGHEST_PROTOCOL)
#     print('save time =' ,time.time()-start_time)

conns=nest.GetConnections(source=node_collections[pop_name][100:102])
# print(type(conns))
# print(type(conns[0]))
for conn in conns:
    print(type(conn))
import multiprocessing as mp
def process_batch(batch):
    try:
        print('starting process batch')
        conns = nest.GetConnections(source=node_collections[pop_name][batch])
        print('Connections queried')
        sources = conns.source
        targets = conns.target
        weights = conns.weight
        print('split')
        
        return (
            np.array(sources),
            np.array(targets),
            np.array(weights)
        )
    except Exception as e:
        print("error in multiprocess: ", e)
batch_size = 20
print(len(pyr_gids))
batches = [pyr_gids[i:np.min([i+batch_size,len(pyr_gids)])] for i in range(0, len(pyr_gids), batch_size)]

start_time=time.time()
rank = nest.Rank()
nprocs = nest.NumProcesses()
results=[]
# for batch in batches:
#     results.append(process_batch(batch))
print('rank:' ,rank)
print('nprocs:' ,nprocs)
n_workers = 4  # adjust to number of cores (or MPI ranks)
with mp.Pool(n_workers) as pool:
    results = pool.map(process_batch, batches)
print('conns querytime =' ,time.time()-start_time)

# start_time=time.time()

# # collect
# sources_all, targets_all, weights_all = map(np.concatenate, zip(*results))
# print('concatenation time =' ,time.time()-start_time)

# start_time=time.time()

# filename = sonata_folder + "/connections_pc.h5"
# with h5py.File(filename, "w") as f:
#     f.create_dataset("pre", data=sources_all, compression="gzip")
#     f.create_dataset("post", data=targets_all, compression="gzip")
#     f.create_dataset("weight", data=weights_all, compression="gzip")

# print('save time =' ,time.time()-start_time)
