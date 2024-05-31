import tensorflow as tf

CHECKPOINT_DIR = "result/checkpoints"

def create_checkpoint_manager(model, optimizer, max_to_keep=5):
    checkpoint = tf.train.Checkpoint(generator1=model['generator1'], 
                                     generator2=model['generator2'], 
                                     discriminator1=model['discriminator1'], 
                                     discriminator2=model['discriminator2'], 
                                     optimizer_G1=optimizer['generator1'], 
                                     optimizer_G2=optimizer['generator2'], 
                                     optimizer_D1=optimizer['discriminator1'], 
                                     optimizer_D2=optimizer['discriminator2'])
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=max_to_keep)
    return checkpoint, checkpoint_manager