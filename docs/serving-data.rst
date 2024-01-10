.. _serving-data:

Serving the converted data
==========================

.. _layouts:

On-disk file layout
-------------------

Neuroglancer expects all chunks from a scale to be located in the same
directory (*flat* layout). This is problematic when working with large volumes,
because filesystems have problems with very large directories. As a result, a
deep layout is used by default when saving the chunks to the filesystem:

- Default deep layout: ``key/x-X/y-Y/z-Z``.

  The default layout is hierarchical, in order to keep the number of directory
  entries to a reasonable number. This means that you need to serve the data
  with URL rewriting.

- Flat layout: ``key/x-X_y-Y_z-Z``

  Chunks are stored in the layout where Neuroglancer will fetch them, so you do
  not need to configure any URL rewriting. Do not use with very large datasets.
  This layout can be used by passing the ``--flat`` option to the conversion
  scripts.


nginx
-----

A Docker image running a specially-configured *nginx* web-server is available
for serving the converted data: `neuroglancer-docker
<https://hub.docker.com/r/ylep/neuroglancer/>`_.

The relevant portion of the nginx configuration is reproduced here:

.. code-block:: nginx

   gzip_static always;
   # All browsers that are compatible with Neuroglancer support gzip encoding
   gunzip      off;

   location ~ ^(.*)/([0-9]+-[0-9]+)_([0-9]+-[0-9]+)_([0-9]+-[0-9]+)$ {
       # Chunks are stored in per-axis sub-directories to prevent
       # having too many files in a single directory
       alias $1/$2/$3/$4;
   }

   location ~ ^(.*):0$ {
       # Microsoft filesystems do not support colons in file names,
       # but they are needed for pre-computed meshes (e.g. 100:0). As
       # :0 is the most common (only?) suffix in use, we look for a
       # file with that suffix stripped.
       try_files $uri $1.json $1 =404;
   }


Apache
------

Alternatively, you serve the pre-computed images using Apache, with the
following Apache configuration (e.g. put it in a ``.htaccess`` file):

.. code-block:: apacheconf

   # If you get a 403 Forbidden error, try to comment out the Options directives
   # below (they may be disallowed by your server's AllowOverride setting).

   <IfModule headers_module>
       # Needed to use the data from a Neuroglancer instance served from a
       # different server (see http://enable-cors.org/server_apache.html).
       Header set Access-Control-Allow-Origin "*"
   </IfModule>

   # Data chunks are stored in sub-directories, in order to avoid having
   # directories with millions of entries. Therefore we need to rewrite URLs
   # because Neuroglancer expects a flat layout.
   Options FollowSymLinks
   RewriteEngine On
   RewriteRule "^(.*)/([0-9]+-[0-9]+)_([0-9]+-[0-9]+)_([0-9]+-[0-9]+)$" "$1/$2/$3/$4"

   # Microsoft filesystems do not support colons in file names, but pre-computed
   # meshes use a colon in the URI (e.g. 100:0). As :0 is the most common (only?)
   # suffix in use, we will serve a file that has this suffix stripped.
   RewriteCond "%{REQUEST_FILENAME}" !-f
   RewriteRule "^(.*):0$" "$1"

   <IfModule mime_module>
       # Allow serving pre-compressed files, which can save a lot of space for raw
       # chunks, compressed segmentation chunks, and mesh chunks.
       #
       # The AddType directive should in theory be replaced by a "RemoveType .gz"
       # directive, but with that configuration Apache fails to serve the
       # pre-compressed chunks (confirmed with Debian version 2.2.22-13+deb7u6).
       # Fixes welcome.
       Options Multiviews
       AddEncoding x-gzip .gz
       AddType application/octet-stream .gz
   </IfModule>


Serving sharded data
====================


Content-Encoding
----------------

Sharded data must be served without any `Content-Encoding header
<https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Encoding>_`.


HTTP Range request
------------------

Sharded data must be served by a webserver that supports `Range header
<https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Range>_`.

For development uses, python's bundled SimpleHTTPServer  `does not support
this <https://github.com/python/cpython/issues/86809>_`. Recommended
alternatives are:

- `http-server (NodeJS)<https://www.npmjs.com/package/http-server>_`

- `RangeHTTPServer(Python) <https://github.com/danvk/RangeHTTPServer>_`

For production uses, most modern static web servers supports range requests.
The below is a list of web servers that were tested and works with sharded
volumes.

- nginx 1.25.3

- httpd 2.4.58

- caddy 2.7.5

In addition, most object storage also supports range requests without
additional configurations.


Enable Access-Control-Allow-Origin header
-----------------------------------------

`Access-Control-Allow-Origin
<https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin>_`
will need to be enabled if the volume is expected to be accessed cross origin.
