from tqdm import tqdm
import tensorflow as  tf
from ops import *
from utils import *
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import misc
from tensorflow.contrib.gan import losses
import random

class Model():
    def __init__(self, checkpoint_dir, mode,loss, input_size=256, batch_size=1, channels=3, output_size=64, lr=0.002,
                 epoch=30, step=20, train_root='train'):
        self.input_size = input_size

        if mode == 'test':
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        self.channels = channels
        self.output_size = output_size
        self.lr = lr
        self.epoch = epoch
        self.step = step
        self.loss_mode=loss
        self.build_model()

        self.checkpoint_dir = checkpoint_dir
        self.train_root = train_root

    def transfer_network(self, inputs):
        with tf.variable_scope('transfer') as scope:
            self.layers = {}

            self.layers['tsf_enc_layer_1'] = LBC(inputs, 64)  # 128

            self.layers['tsf_enc_layer_2'] = LBC(self.layers['tsf_enc_layer_1'], 128)  # 64

            self.layers['tsf_enc_layer_3'] = LBC(self.layers['tsf_enc_layer_2'], 256)  # 32

            self.layers['tsf_enc_layer_4'] = LBC(self.layers['tsf_enc_layer_3'], 512)  # 16

            self.layers['tsf_enc_layer_5'] = LBC(self.layers['tsf_enc_layer_4'], 512)  # 8

            self.layers['tsf_enc_layer_6'] = LBC(self.layers['tsf_enc_layer_5'], 512)  # 4

            res_block_1 = self.resnet(self.layers['tsf_enc_layer_6'])

            self.layers['tsf_enc_layer_7'] = LBC(self.layers['tsf_enc_layer_6'], 512)  # 2
            res_block_2 = self.resnet(self.layers['tsf_enc_layer_7'])

            self.layers['tsf_latent_feature'] = LBC(self.layers['tsf_enc_layer_7'], 512)  # 1

            self.layers['tsf_dec_layers_7'] = RBD(self.layers['tsf_latent_feature'] , 512)  # 2

            self.layers['tsf_dec_layers_6'] = RBD(self.layers['tsf_dec_layers_7']+ res_block_2, 512)  # 4

            self.layers['tsf_dec_layers_5'] = RBD(self.layers['tsf_dec_layers_6']+res_block_1, 512)  # 8

            self.layers['tsf_dec_layers_4'] = RBD(self.layers['tsf_dec_layers_5'], output_size=512)

            self.layers['tsf_dec_layers_3'] = RBD(self.layers['tsf_dec_layers_4'], output_size=256)

            self.layers['tsf_dec_layers_2'] = RBD(self.layers['tsf_dec_layers_3'], output_size=128)

            self.layers['tsf_dec_layers_1'] = RBD(self.layers['tsf_dec_layers_2'], output_size=64)

            self.layers['tsf_output'] = deconv2d(self.layers['tsf_dec_layers_1'], 1)

            return tf.nn.tanh(self.layers['tsf_output'])

    def discriminator(self, input, target, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()

            output_size = 64
            # target = target+ tf.random.normal(shape=tf.shape(target), mean=0.0, stddev=0.1)
            image = tf.concat([input,target],3)



            L1 = RBC(image,64)
            L2 = RBC(input=L1, output_size=128)
            L3 = RBC(input=L2, output_size=256)
            L4 = RBC(input=L3, output_size=512)

            L5 = RBC(input=L4, output_size=512)

            L6 = RBC(input=L5, output_size=512)

            L7 = RBC(input=L6, output_size=512)

            L8 = tf.layers.dense(tf.reshape(L4,[self.batch_size,-1]),1)

            # n_layers = 3
            # layers = []
            #
            # # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            # input = tf.concat([input, target], axis=3)
            #
            # # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            # with tf.variable_scope("layer_1"):
            #     convolved = discrim_conv(input, 64, stride=2)
            #     rectified = leaky_relu(convolved, 0.2)
            #     layers.append(rectified)
            #
            # # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            # # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            # # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            # for i in range(n_layers):
            #     with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            #         out_channels = 64 * min(2**(i+1), 8)
            #         stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            #         convolved = discrim_conv(layers[-1], out_channels, stride=stride)
            #         normalized = batch_norm(convolved)
            #         rectified = leaky_relu(normalized, 0.2)
            #         layers.append(rectified)
            #
            # # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            # with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            #     convolved = discrim_conv(rectified, out_channels=1, stride=1)
            #     layers.append(convolved)

        return L8

    def supervised_network(self, inputs):
        with tf.variable_scope('supervised') as scope:
            self.layers['spv_enc_layer_1'] = LBC(inputs, 64)  # (128 , 128 , 128)

            self.layers['spv_enc_layer_2'] = LBC(self.layers['spv_enc_layer_1'], 128)  # (64 , 64 128,)

            self.layers['spv_enc_layer_3'] = LBC(self.layers['spv_enc_layer_2'], 256)  # (32 , 32 256)

            self.layers['spv_enc_layer_4'] = LBC(self.layers['spv_enc_layer_3'], 512)  # (16 , 16 , 512)

            self.layers['spv_enc_layer_5'] = LBC(self.layers['spv_enc_layer_4'], 512)  # (8 , 8 , 512)

            self.layers['spv_enc_layer_6'] = LBC(self.layers['spv_enc_layer_5'], 512)  # 4 ,4

            self.layers['spv_enc_layer_7'] = LBC(self.layers['spv_enc_layer_6'], 512)  # 2, 2

            self.layers['spv_latent_feature'] = LBC(self.layers['spv_enc_layer_7'], 512)  # 1,1

            self.layers['spv_dec_layers_7'] = RBD(self.layers['spv_latent_feature'],  # 2,2
                                                  output_size=512)

            self.layers['spv_dec_layers_6'] = RBD(
                tf.concat([self.layers['spv_dec_layers_7'], self.layers['spv_enc_layer_7']], axis=3),
                # 4,4
                output_size=512)

            self.layers['spv_dec_layers_5'] = RBD(
                tf.concat([self.layers['spv_dec_layers_6'], self.layers['spv_enc_layer_6']], axis=3),  # 8 , 8
                output_size=512)

            self.layers['spv_dec_layers_4'] = RBD(
                tf.concat([self.layers['spv_dec_layers_5'], self.layers['spv_enc_layer_5']], axis=3),  # 16 , 16
                output_size=512)

            self.layers['spv_dec_layers_3'] = RBD(
                tf.concat([self.layers['spv_dec_layers_4'], self.layers['spv_enc_layer_4']], axis=3),
                output_size=256)

            self.layers['spv_dec_layers_2'] = RBD(
                tf.concat([self.layers['spv_dec_layers_3'], self.layers['spv_enc_layer_3']], axis=3),
                output_size=128)

            self.layers['spv_dec_layers_1'] = RBD(
                tf.concat([self.layers['spv_dec_layers_2'], self.layers['spv_enc_layer_2']], axis=3),
                output_size=64)

            self.layers['spv_output'] = deconv2d(self.layers['spv_dec_layers_1'], 1)

            return self.layers['spv_output']

    def build_model(self):

        self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size, self.input_size, 1])
        # x is standard font image x

        self.y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size, self.input_size, 1])
        # y is calligraphy image y
        self.generate = self.transfer_network(self.x)

        self.encoded = self.supervised_network(inputs=self.y)

        self.fake = self.discriminator(input=self.x, target=self.generate, reuse=False)

        self.real = self.discriminator(input=self.x, target=self.y, reuse=True)

        lambda_s = 100
        lambda_r = 100
        EPS = 1e-10


        self.L_supervise = lambda_s * tf.reduce_mean(tf.abs((self.y - self.encoded)))
        self.g_l1 = lambda_r * tf.reduce_mean(tf.abs((self.y - self.generate)))

        self.L_reconstruct = []

        for i in range(1, 5):
            self.L_reconstruct.append(tf.reduce_mean(
                tf.abs(self.layers['tsf_dec_layers_%d' % i] - self.layers['spv_dec_layers_%d' % i]), axis=[1, 2, 3]))
        self.L_reconstruct = tf.reduce_mean(tf.reduce_sum(self.L_reconstruct, 0)) * lambda_r

        if self.loss_mode == 'vanila':
            self.L_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real, labels=tf.ones_like(self.real)))
            self.L_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake, labels=tf.zeros_like(self.fake)))
            self.L_adversarial = self.L_fake + self.L_real

            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake, labels=tf.ones_like(self.fake)))

            #
            # self.g_loss = -tf.reduce_mean(tf.log(self.fake + EPS))
            # self.L_adversarial = -tf.reduce_mean(tf.log(self.real+EPS) + tf.log(1-self.fake + EPS))

        if self.loss_mode == 'lsgan':
            self.L_real = self.criterionGAN(self.real, tf.ones_like(self.real))
            self.L_fake = self.criterionGAN(self.fake, tf.zeros_like(self.fake))
            self.g_loss = self.criterionGAN(self.fake, tf.ones_like(self.fake))
            self.L_adversarial = self.L_real + self.L_fake

        print('loss mode : ', self.loss_mode)
        #
        # self.L_adversarial = losses.wargs.modified_discriminator_loss(self.real_logit,self.fake_logit)
        # self.g_loss = losses.wargs.modified_generator_loss(self.fake)
        #
        # self.L_real = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real, labels=tf.ones_like(self.real)))
        # self.L_fake = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake, labels=tf.zeros_like(self.fake)))
        # self.L_adversarial = self.L_real + self.L_fake
        #
        # self.g_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake, labels=tf.ones_like(self.fake)))
        transfer_grad_vars = [var for var in tf.trainable_variables() if 'transfer' in var.name]
        supervised_grad_vars = [var for var in tf.trainable_variables() if 'supervised' in var.name]
        discriminator_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]

        self.g_loss_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.g_l1 + self.L_reconstruct +self.g_loss, var_list=transfer_grad_vars)
        self.L_supervise_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.L_supervise,
                                                                                      var_list=supervised_grad_vars)
        # self.gan_opt = tf.test.AdamOptimizer(learning_rate=self.lr).minimize(self.g_loss,var_list=transfer_grad_vars)
        self.adversarial_loss_opt = tf.train.AdamOptimizer(learning_rate=0.002).minimize(self.L_adversarial,
                                                                                          var_list=discriminator_vars)
        # self.L_reconstruct_opt = tf.test.AdamOptimizer(learning_rate=self.lr).minimize(self.L_reconstruct,var_list=transfer_grad_vars)


    def main(self):
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        print('checkpoint file : ', checkpoint)
        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())
            # saver.restore(sess=sess,save_path=checkpoint)

            print "all variables intialized"
            train_original = os.path.join(self.train_root,'original')
            train_target = os.path.join(self.train_root, 'target')
            print "start training"
            imgs = os.listdir(train_original)
            path_x = [os.path.join(train_original,img) for img in imgs]
            path_y = [os.path.join(train_target,img) for img in imgs]

            path = [[path_x[i], path_y[i]] for i in range(len(path_x))]
            steps = len(path) / self.batch_size
            for epoch in tqdm(range(self.epoch)):
                random.shuffle(path)
                for step in range(steps):


                    x, y = create_batches(path[step * self.batch_size: step * self.batch_size + self.batch_size])

                    g_loss, L_reconstruct, g_l1, _ = sess.run(
                        [self.g_loss, self.L_reconstruct, self.g_l1, self.g_loss_opt], feed_dict={self.x: x, self.y: y})

                    L1_loss, _ = sess.run([self.L_supervise, self.L_supervise_opt], feed_dict={self.x: x, self.y: y})

                    real, fake = sess.run([self.real, self.fake], feed_dict={self.x: x, self.y: y})
                    #
                    if step % 10 == 0:
                        adversarial_loss, _ = sess.run([self.L_adversarial, self.adversarial_loss_opt],
                                                       feed_dict={self.x: x, self.y: y})
                        print(
                            "epoch {} : step : {} l1_loss {} g_loss :{:.4} g_l1 :{:.4} reconstruct loss {:.4} adv {:.4} real :{:.4} fake :{:.4}").format(
                            epoch, step,L1_loss, g_loss, g_l1,  L_reconstruct, adversarial_loss, np.mean(real),
                            np.mean(fake))



                saver.save(sess, self.checkpoint_dir + 'model', global_step=epoch)

    def test(self,ckpt,test_data,result_dir):
        if ckpt is None:
            checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        print('checkpoint file : ', checkpoint)

        saver = tf.train.Saver()
        img_list= os.listdir(test_data)
        with tf.Session() as sess:
            saver.restore(sess=sess, save_path=checkpoint)
            for img in img_list:
                x=[load_images(os.path.join(test_data,img))]
                generated = sess.run(self.generate, feed_dict={self.x: x})
                generate = generated[0].reshape(256, 256)
                if not os.path.exists(result_dir):
                    os.mkdir(result_dir)
                misc.imsave(os.path.join(result_dir,img),generate)

    def criterionGAN(self,input,target):
        return tf.reduce_mean((input-target)**2)

    def resnet(self, x):
        _, _, _, channels = x.get_shape().as_list()
        net = conv2d(x, channels, kernel=[3, 3], stride=[1, 1])
        net = batch_norm(net)
        net = relu(net)

        net = conv2d(net, channels, kernel=[3, 3], stride=[1, 1])
        net = batch_norm(net)
        net = relu(net)

        return x + net

