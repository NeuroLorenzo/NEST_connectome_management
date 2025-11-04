import nest


nest.set_verbosity("M_ALL")
nest.ResetKernel()
dt=.1
nest.SetKernelStatus({""
                        "local_num_threads": 4,
                        "print_time" : True,
	                    "rng_seed": 12345,
	                    "resolution": dt
                        
                        })


neurons=nest.Create('ht_neuron',200)
nest.Connect(neurons,neurons,{'rule':'all_to_all'},{'synapse_model':'stdp_synapse','weight':1,'delay':1,'receptor_type':1})
sg=nest.Create('spike_generator',{'spike_times':[10,20,30,40]})
nest.Connect(sg,neurons[:10],{'rule':'all_to_all'},{'synapse_model':'static_synapse','weight':5,'delay':1,'receptor_type':1})
sr=nest.Create('spike_recorder')
nest.Connect(neurons,sr)
nest.Simulate(100)

print(nest.GetStatus(sr,'events'))
