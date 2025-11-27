import nest
import numpy as np
import time
nest.SetKernelStatus({
                        "local_num_threads": 1,
                        # "local_num_threads": 24,
                     })


neurons=nest.Create('ht_neuron',10000)
for i in range(4000):
    nest.Connect(neurons[i],neurons[i+100],{'rule':'all_to_all'},{'synapse_model':'stdp_synapse','weight':2,'delay':1,'receptor_type':1})
    # nest.Connect(neurons[i],neurons[i+100],{'rule':'all_to_all'},{'synapse_model':'stdp_synapse','weight':i/10,'delay':1,'receptor_type':1})

pre_ids =np.arange(0,3999)
post_ids=pre_ids+100
weights=pre_ids/10

# start=time.time()
# conns=nest.GetConnections(source=neurons)
# end=time.time()
# print("Get connections:", end-start)

def test_set(pre,post,weights):
    conns=nest.GetConnections(source=pre)

    sources = nest.GetStatus(conns, 'source')
    print(sources)
    targets = nest.GetStatus(conns, 'target')
    print(targets)
    mask = np.isin(pre,sources)
    saved_targ = post[mask]
    index_mapping ={}
    for index,value in enumerate(targets):
        if value in index_mapping:
            index_mapping[value].append(index)
        else:
            index_mapping[value] = [index]

    found_index = np.empty_like(saved_targ, dtype=int)
    for i, value in enumerate(saved_targ):
        idx = index_mapping[value].pop(0)
        found_index[idx] = i

    #print(idp,source[0], len(targ),len(ww[pre==idp]), len(found_index) )
    conns.set({'weight':weights[mask][found_index]})
    
test_set(pre_ids,post_ids,weights)
print("Connections created")

nest.Simulate(100)
print("Connections created")


start=time.time()
conns=nest.GetConnections(source=neurons)
end=time.time()
print(end-start)

# print(conns.sources)
# print(conns.sources())
# for con_s in conns.get("source"):  # sources():
#     n=1
    # print(con_s)
# print(dir(conns._datum[0]))
# conns_sources=conns.get(['source'])
# conns.print_full = True
# print(conns)
end2=time.time()
print(end2-end)
print(type(conns))
conns_sources=conns.get('source')
conns_tagets=conns.get('target')
# print(conns_sources)
# print(conns_tagets)
print('done',time.time()-end2)
# print('queried ',len(conns.get(['source'])['source']), ' connections in: ', end-start,'ms')
# print(np.unique(conns.get(['source'])['source']).shape)
# print(np.unique(conns.get(['target'])['target']).shape)
