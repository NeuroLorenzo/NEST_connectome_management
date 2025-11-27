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

pre =np.arange(0,4000)
post=pre+100
weights=pre/10

# start=time.time()
# conns=nest.GetConnections(source=neurons)
# end=time.time()
# print("Get connections:", end-start)

def test_set(pre,post,weights):
    t1=time.time()
    conns=nest.GetConnections(source=neurons[pre])
    t2=time.time()
    print('getConnections time: ',t2-t1)
    sources = nest.GetStatus(conns, 'source')
    t3=time.time()
    print('get status src time: ',t3-t2)
    # print(sources)
    targets = nest.GetStatus(conns, 'target')
    t4=time.time()
    print('get status trg time: ',t4-t3)
    # print(targets)
    mask = np.isin(pre+1,sources)
    saved_targ = post[mask]+1
    index_mapping ={}
    for index,value in enumerate(targets):
        if value in index_mapping:
            index_mapping[value].append(index)
        else:
            index_mapping[value] = [index]
    t5=time.time()
    print('index mapping time: ',t5-t4)

    found_index = np.empty_like(saved_targ, dtype=int)
    for i, value in enumerate(saved_targ):
        idx = index_mapping[value].pop(0)
        found_index[idx] = i
    t6=time.time()
    print('found index time: ',t6-t5)

    print(weights.shape)
    print(weights[mask].shape)
    print(found_index.shape)
    #print(idp,source[0], len(targ),len(ww[pre==idp]), len(found_index) )
    conns.set({'weight':weights[mask][found_index]})
    t7=time.time()
    print('conns set time: ',t7-t6)
    
test_set(pre,post,weights)
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
conns_weights=conns.get('weight')
print('done',time.time()-end2)

print('sources:',conns_sources[:10])
print('targets:',conns_tagets[:10])
print('weights:',conns_weights[:10])
print('conns:',conns)
