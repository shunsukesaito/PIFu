#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

from lib.renderer.camera import Camera
import numpy as np
from lib.renderer.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl
from lib.renderer.camera import Camera
import os
import cv2
import time
import math
import random
import pyexr
import argparse
from tqdm import tqdm


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def rotateSH(SH, R):
    SHn = SH
    
    # 1st order
    SHn[1] = R[1,1]*SH[1] - R[1,2]*SH[2] + R[1,0]*SH[3]
    SHn[2] = -R[2,1]*SH[1] + R[2,2]*SH[2] - R[2,0]*SH[3]
    SHn[3] = R[0,1]*SH[1] - R[0,2]*SH[2] + R[0,0]*SH[3]

    # 2nd order
    SHn[4:,0] = rotateBand2(SH[4:,0],R)
    SHn[4:,1] = rotateBand2(SH[4:,1],R)
    SHn[4:,2] = rotateBand2(SH[4:,2],R)

    return SHn

def rotateBand2(x, R):
    s_c3 = 0.94617469575
    s_c4 = -0.31539156525
    s_c5 = 0.54627421529

    s_c_scale = 1.0/0.91529123286551084
    s_c_scale_inv = 0.91529123286551084

    s_rc2 = 1.5853309190550713*s_c_scale
    s_c4_div_c3 = s_c4/s_c3
    s_c4_div_c3_x2 = (s_c4/s_c3)*2.0

    s_scale_dst2 = s_c3 * s_c_scale_inv
    s_scale_dst4 = s_c5 * s_c_scale_inv

    sh0 =  x[3] + x[4] + x[4] - x[1]
    sh1 =  x[0] + s_rc2*x[2] +  x[3] + x[4]
    sh2 =  x[0]
    sh3 = -x[3]
    sh4 = -x[1]
    
    r2x = R[0][0] + R[0][1]
    r2y = R[1][0] + R[1][1]
    r2z = R[2][0] + R[2][1]
    
    r3x = R[0][0] + R[0][2]
    r3y = R[1][0] + R[1][2]
    r3z = R[2][0] + R[2][2]
    
    r4x = R[0][1] + R[0][2]
    r4y = R[1][1] + R[1][2]
    r4z = R[2][1] + R[2][2]
    
    sh0_x = sh0 * R[0][0]
    sh0_y = sh0 * R[1][0]
    d0 = sh0_x * R[1][0]
    d1 = sh0_y * R[2][0]
    d2 = sh0 * (R[2][0] * R[2][0] + s_c4_div_c3)
    d3 = sh0_x * R[2][0]
    d4 = sh0_x * R[0][0] - sh0_y * R[1][0]
    
    sh1_x = sh1 * R[0][2]
    sh1_y = sh1 * R[1][2]
    d0 += sh1_x * R[1][2]
    d1 += sh1_y * R[2][2]
    d2 += sh1 * (R[2][2] * R[2][2] + s_c4_div_c3)
    d3 += sh1_x * R[2][2]
    d4 += sh1_x * R[0][2] - sh1_y * R[1][2]
    
    sh2_x = sh2 * r2x
    sh2_y = sh2 * r2y
    d0 += sh2_x * r2y
    d1 += sh2_y * r2z
    d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2)
    d3 += sh2_x * r2z
    d4 += sh2_x * r2x - sh2_y * r2y
    
    sh3_x = sh3 * r3x
    sh3_y = sh3 * r3y
    d0 += sh3_x * r3y
    d1 += sh3_y * r3z
    d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2)
    d3 += sh3_x * r3z
    d4 += sh3_x * r3x - sh3_y * r3y
    
    sh4_x = sh4 * r4x
    sh4_y = sh4 * r4y
    d0 += sh4_x * r4y
    d1 += sh4_y * r4z
    d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2)
    d3 += sh4_x * r4z
    d4 += sh4_x * r4x - sh4_y * r4y

    dst = x
    dst[0] = d0
    dst[1] = -d1
    dst[2] = d2 * s_scale_dst2
    dst[3] = -d3
    dst[4] = d4 * s_scale_dst4

    return dst

def render_prt_ortho(out_path, folder_name, subject_name, shs, rndr, rndr_uv, im_size, angl_step=4, n_light=1, pitch=[0]):
    cam = Camera(width=im_size, height=im_size)
    cam.ortho_ratio = 0.4 * (512 / im_size)
    cam.near = -100
    cam.far = 100
    cam.sanity_check()

    # set path for obj, prt
    mesh_file = os.path.join(folder_name, subject_name + '_100k.obj')
    if not os.path.exists(mesh_file):
        print('ERROR: obj file does not exist!!', mesh_file)
        return 
    prt_file = os.path.join(folder_name, 'bounce', 'bounce0.txt')
    if not os.path.exists(prt_file):
        print('ERROR: prt file does not exist!!!', prt_file)
        return
    face_prt_file = os.path.join(folder_name, 'bounce', 'face.npy')
    if not os.path.exists(face_prt_file):
        print('ERROR: face prt file does not exist!!!', prt_file)
        return
    text_file = os.path.join(folder_name, 'tex', subject_name + '_dif_2k.jpg')
    if not os.path.exists(text_file):
        print('ERROR: dif file does not exist!!', text_file)
        return             

    texture_image = cv2.imread(text_file)
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    vertices, faces, normals, faces_normals, textures, face_textures = load_obj_mesh(mesh_file, with_normal=True, with_texture=True)
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
    
    vmed = np.median(vertices, 0)
    vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
    y_scale = 180/(vmax[up_axis] - vmin[up_axis])

    rndr.set_norm_mat(y_scale, vmed)
    rndr_uv.set_norm_mat(y_scale, vmed)

    tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
    prt = np.loadtxt(prt_file)
    face_prt = np.load(face_prt_file)
    rndr.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)    
    rndr.set_albedo(texture_image)

    rndr_uv.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)   
    rndr_uv.set_albedo(texture_image)

    os.makedirs(os.path.join(out_path, 'GEO', 'OBJ', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'PARAM', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'RENDER', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'MASK', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'UV_RENDER', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'UV_MASK', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'UV_POS', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'UV_NORMAL', subject_name),exist_ok=True)

    if not os.path.exists(os.path.join(out_path, 'val.txt')):
        f = open(os.path.join(out_path, 'val.txt'), 'w')
        f.close()

    # copy obj file
    cmd = 'cp %s %s' % (mesh_file, os.path.join(out_path, 'GEO', 'OBJ', subject_name))
    print(cmd)
    os.system(cmd)

    for p in pitch:
        for y in tqdm(range(0, 360, angl_step)):
            R = np.matmul(make_rotate(math.radians(p), 0, 0), make_rotate(0, math.radians(y), 0))
            if up_axis == 2:
                R = np.matmul(R, make_rotate(math.radians(90),0,0))

            rndr.rot_matrix = R
            rndr_uv.rot_matrix = R
            rndr.set_camera(cam)
            rndr_uv.set_camera(cam)

            for j in range(n_light):
                sh_id = random.randint(0,shs.shape[0]-1)
                sh = shs[sh_id]
                sh_angle = 0.2*np.pi*(random.random()-0.5)
                sh = rotateSH(sh, make_rotate(0, sh_angle, 0).T)

                dic = {'sh': sh, 'ortho_ratio': cam.ortho_ratio, 'scale': y_scale, 'center': vmed, 'R': R}
                
                rndr.set_sh(sh)        
                rndr.analytic = False
                rndr.use_inverse_depth = False
                rndr.display()

                out_all_f = rndr.get_color(0)
                out_mask = out_all_f[:,:,3]
                out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGR)

                np.save(os.path.join(out_path, 'PARAM', subject_name, '%d_%d_%02d.npy'%(y,p,j)),dic)
                cv2.imwrite(os.path.join(out_path, 'RENDER', subject_name, '%d_%d_%02d.jpg'%(y,p,j)),255.0*out_all_f)
                cv2.imwrite(os.path.join(out_path, 'MASK', subject_name, '%d_%d_%02d.png'%(y,p,j)),255.0*out_mask)

                rndr_uv.set_sh(sh)
                rndr_uv.analytic = False
                rndr_uv.use_inverse_depth = False
                rndr_uv.display()

                uv_color = rndr_uv.get_color(0)
                uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(os.path.join(out_path, 'UV_RENDER', subject_name, '%d_%d_%02d.jpg'%(y,p,j)),255.0*uv_color)

                if y == 0 and j == 0 and p == pitch[0]:
                    uv_pos = rndr_uv.get_color(1)
                    uv_mask = uv_pos[:,:,3]
                    cv2.imwrite(os.path.join(out_path, 'UV_MASK', subject_name, '00.png'),255.0*uv_mask)

                    data = {'default': uv_pos[:,:,:3]} # default is a reserved name
                    pyexr.write(os.path.join(out_path, 'UV_POS', subject_name, '00.exr'), data) 

                    uv_nml = rndr_uv.get_color(2)
                    uv_nml = cv2.cvtColor(uv_nml, cv2.COLOR_RGBA2BGR)
                    cv2.imwrite(os.path.join(out_path, 'UV_NORMAL', subject_name, '00.png'),255.0*uv_nml)


if __name__ == '__main__':
    shs = np.load('./env_sh.npy')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/home/shunsuke/Downloads/rp_dennis_posed_004_OBJ')
    parser.add_argument('-o', '--out_dir', type=str, default='/home/shunsuke/Documents/hf_human')
    parser.add_argument('-m', '--ms_rate', type=int, default=1, help='higher ms rate results in less aliased output. MESA renderer only supports ms_rate=1.')
    parser.add_argument('-e', '--egl',  action='store_true', help='egl rendering option. use this when rendering with headless server with NVIDIA GPU')
    parser.add_argument('-s', '--size',  type=int, default=512, help='rendering image size')
    args = parser.parse_args()

    # NOTE: GL context has to be created before any other OpenGL function loads.
    from lib.renderer.gl.init_gl import initialize_GL_context
    initialize_GL_context(width=args.size, height=args.size, egl=args.egl)

    from lib.renderer.gl.prt_render import PRTRender
    rndr = PRTRender(width=args.size, height=args.size, ms_rate=args.ms_rate, egl=args.egl)
    rndr_uv = PRTRender(width=args.size, height=args.size, uv_mode=True, egl=args.egl)

    if args.input[-1] == '/':
        args.input = args.input[:-1]
    subject_name = args.input.split('/')[-1][:-4]
    render_prt_ortho(args.out_dir, args.input, subject_name, shs, rndr, rndr_uv, args.size, 1, 1, pitch=[0])